"""Exclude yes/no and binary MMAR questions from the freeform question pool.

Two passes over ``MMAR-meta.jsonl``:

1. ``is_yes_no_question`` tags items whose options are Yes/No, or that have
   two options and a polar stem ("is this", "does the", ...).
2. Gemini reads the question text only (no gold answer, no model answers, no
   choices) for every remaining item and decides whether the wording itself
   implies a two-way answer, e.g. "in front of or behind".

Writes into ``--out-dir`` (default: this folder):

- ``gemini_binary_judgments.jsonl`` — full judge rows (resumable)
- ``gemini_binary_question_ids.csv`` — binary IDs found by Gemini
- ``open_ended_question_ids.csv`` — the remaining open-ended pool
- ``gemini_binary_filter_summary.json`` — counts

Usage::

    export GEMINI_API_KEY=...

    uv run python answer-variety/filter_binary_questions.py
    uv run python answer-variety/filter_binary_questions.py --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmar_common import (  # noqa: E402
    DEFAULT_DATA_DIR,
    THINK_BLOCK_RE,
    load_jsonl,
    write_json,
    write_jsonl,
)

DEFAULT_META = DEFAULT_DATA_DIR / "MMAR-meta.jsonl"
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
DEFAULT_LIMIT = 0  # 0 = all questions
JUDGMENTS_PATH = PACKAGE_DIR / "gemini_binary_judgments.jsonl"
GEMINI_BINARY_IDS_PATH = PACKAGE_DIR / "gemini_binary_question_ids.csv"
OPEN_ENDED_IDS_PATH = PACKAGE_DIR / "open_ended_question_ids.csv"
SUMMARY_PATH = PACKAGE_DIR / "gemini_binary_filter_summary.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic yes/no detection
# ---------------------------------------------------------------------------

_CHOICE_LETTER_RE = re.compile(r"^[(\[]?[A-Da-d][)\].:]+\s*")
_YES_CHOICE_RE = re.compile(r"^(yes|true|correct)\b", re.IGNORECASE)
_NO_CHOICE_RE = re.compile(r"^(no|false|incorrect)\b", re.IGNORECASE)
# Polar 2-choice stems: AUX + determiner/pronoun.
YES_NO_QUESTION_PREFIXES = (
    "is this",
    "is the",
    "is that",
    "is there",
    "does this",
    "does the",
    "does that",
    "did this",
    "did the",
    "are the",
    "are they",
    "are these",
    "was the",
    "was this",
)
_YES_NO_STEM_RE = re.compile(
    r"(?:^|[.?!]\s+)(?:"
    + "|".join(re.escape(prefix) for prefix in YES_NO_QUESTION_PREFIXES)
    + r")\b",
    re.IGNORECASE,
)


def _normalize_choice(choice: str) -> str:
    text = str(choice or "").strip().strip("\"'")
    return _CHOICE_LETTER_RE.sub("", text).strip()


def _choice_polarity(choice: str) -> str | None:
    """Return ``yes``, ``no``, or None if the choice is not a yes/no string."""
    text = _normalize_choice(choice)
    if not text:
        return None
    if _YES_CHOICE_RE.match(text):
        return "yes"
    if _NO_CHOICE_RE.match(text):
        return "no"
    return None


def _choices_are_yes_no(choices: list | None) -> bool:
    labels = [_choice_polarity(c) for c in (choices or []) if str(c).strip()]
    if len(labels) < 2 or any(label is None for label in labels):
        return False
    return "yes" in labels and "no" in labels


def _question_has_polar_stem(question: str | None) -> bool:
    text = str(question or "").strip()
    if not text:
        return False
    return _YES_NO_STEM_RE.search(text) is not None


def is_yes_no_question(
    choices: list | None, question: str | None = None
) -> bool:
    """True for yes/no-style items that collapse freeform variety.

    1. Every MCQ option is a Yes/No (or True/False) polarity string,
       including prefixed forms such as ``Yes, he complied.``
    2. Or there are exactly two choices and the stem is polar
       (``is this``, ``is the``, ``does this``, ``does the``, plus
       matching variants like ``is that``, ``did the``, ``are they``).
    """
    if _choices_are_yes_no(choices):
        return True
    nonempty = [c for c in (choices or []) if str(c).strip()]
    return len(nonempty) == 2 and _question_has_polar_stem(question)


# ---------------------------------------------------------------------------
# Gemini verdict parsing
# ---------------------------------------------------------------------------

_VERDICT_PHRASE_RE = re.compile(
    r"(?:final\s+)?(?:verdict|answer|identical)\s*[:\-]?\s*[\"']?(yes|no)[\"']?\b",
    re.IGNORECASE,
)
_BARE_VERDICT_RE = re.compile(r"^[\"']?(yes|no)[\"']?[.\s]*$", re.IGNORECASE)


def _norm_verdict(token: str) -> str:
    return "Yes" if token.lower() == "yes" else "No"


def visible_judge_text(text: str) -> str:
    raw = str(text or "")
    match = THINK_BLOCK_RE.search(raw)
    if not match:
        return raw.strip()
    remainder = (raw[: match.start()] + raw[match.end() :]).strip()
    return remainder or raw.strip()


def parse_yes_no_verdict(text: str) -> str | None:
    """Return ``Yes`` / ``No`` from a judge reply, or None if unparseable."""
    region = visible_judge_text(text)
    if not region:
        return None
    matches = list(_VERDICT_PHRASE_RE.finditer(region))
    if matches:
        return _norm_verdict(matches[-1].group(1))
    lines = [line.strip() for line in region.splitlines() if line.strip()]
    for line in reversed(lines[-12:]):
        bare = _BARE_VERDICT_RE.match(line)
        if bare:
            return _norm_verdict(bare.group(1))
        phrase = _VERDICT_PHRASE_RE.search(line)
        if phrase:
            return _norm_verdict(phrase.group(1))
    return None


def extract_reason(text: str, verdict: str | None) -> str:
    """Drop the trailing Yes/No line from the judge reply."""
    region = visible_judge_text(text)
    if not region:
        return ""
    lines = region.splitlines()
    if verdict and lines:
        last = lines[-1].strip().strip("\"'")
        if last.lower() == verdict.lower():
            return "\n".join(lines[:-1]).strip()
    return region.strip()


# ---------------------------------------------------------------------------
# Resumable judgments file
# ---------------------------------------------------------------------------


def _is_fully_graded(item: dict) -> bool:
    if not item.get("id"):
        return False
    if item.get("verdict") not in {"Yes", "No"}:
        return False
    return bool(str(item.get("raw_response") or "").strip())


def load_completed_ids(path: Path) -> set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    return {str(item["id"]) for item in load_jsonl(path) if _is_fully_graded(item)}


def prune_incomplete_judgments(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    records = load_jsonl(path)
    keep = [item for item in records if _is_fully_graded(item)]
    removed = len(records) - len(keep)
    if removed <= 0:
        return 0
    by_id: dict[str, dict] = {str(item["id"]): item for item in keep}
    write_jsonl(path, list(by_id.values()), mode="w")
    return removed


def write_id_csv(path: Path, ids: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id"])
        for qid in ids:
            writer.writerow([qid])
    return path


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------


def _gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            piece = getattr(part, "text", None)
            if piece:
                parts.append(str(piece))
    return "\n".join(parts).strip()


class GeminiTextClassifier:
    """Text-only Gemini call with QPS throttling and rate-limit retries."""

    def __init__(
        self,
        *,
        model_name: str,
        system_instruction: str,
        api_max_retries: int = 20,
        api_retry_interval: float = 1.0,
        qps: float = 4.0,
        timeout: float = 180.0,
        max_output_tokens: int = 2048,
        thinking_level: str = "medium",
    ):
        from google import genai
        from google.genai import types

        api_key = (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        ).strip()
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY to call Gemini.")
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.api_max_retries = api_max_retries
        self.api_retry_interval = api_retry_interval
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self._next_time = 0.0
        self._interval = 1.0 / max(float(qps), 0.1)
        self._lock = asyncio.Lock()

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (429, 418):
            return True
        text = str(exc).lower()
        return "rate" in text or "resource exhausted" in text or "429" in text

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self._interval

    async def _complete(self, user_prompt: str) -> str:
        thinking = None
        if self.thinking_level:
            thinking = self.types.ThinkingConfig(thinking_level=self.thinking_level)
        config = self.types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            max_output_tokens=int(self.max_output_tokens),
            thinking_config=thinking,
        )
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[user_prompt],
            config=config,
        )
        return _gemini_response_text(response)

    async def call(self, log_id: str, user_prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.api_max_retries + 1):
            try:
                await self._throttle()
                text = await asyncio.wait_for(
                    self._complete(user_prompt),
                    timeout=self.timeout,
                )
            except TimeoutError as exc:
                last_exc = exc
                logger.warning("%s (api_attempt=%s) timeout: %s", log_id, attempt, exc)
                await asyncio.sleep(self.api_retry_interval)
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if self._is_rate_limit(exc):
                    logger.warning(
                        "%s (api_attempt=%s) rate limit: %s", log_id, attempt, exc
                    )
                    await asyncio.sleep(self.api_retry_interval * attempt)
                    continue
                raise
            if text:
                return text
            logger.warning("%s (api_attempt=%s) empty Gemini response", log_id, attempt)
            await asyncio.sleep(self.api_retry_interval)
        raise RuntimeError(
            f"{log_id} Gemini retries exhausted ({self.api_max_retries}): {last_exc}"
        )


# ---------------------------------------------------------------------------
# Binary-question filter
# ---------------------------------------------------------------------------

BINARY_SYSTEM_PROMPT = (
    "You classify audio-understanding questions from question text alone.\n"
    "Decide whether the wording implies a binary answer space: exactly two "
    "mutually exclusive values.\n"
    "\n"
    "Binary examples: yes/no; true/false; presence vs absence of one "
    "attribute; two named alternatives such as \"in front of or behind the "
    "listener\", \"male or female\", \"arriving or departing\", "
    "\"towards or away\", \"major or minor\".\n"
    "\n"
    "Not binary when many answers are plausible: what/which/who/how/why "
    "with an open set, counts, categories, transcription, or description. "
    "A question that lists three or more alternatives is not binary.\n"
    "\n"
    "Do not assume multiple-choice options or a gold answer. Judge only "
    "the question wording.\n"
    "\n"
    "Reason briefly if needed, then end your reply with a single final line "
    "containing only Yes or No. Yes means the question implies a binary "
    "answer. No means it is more open-ended."
)


def create_binary_user_prompt(question: str) -> str:
    text = str(question or "").strip() or "(empty question)"
    return (
        "Does this question imply a binary answer?\n\n"
        f"Question:\n{text}"
    )


def load_mmar_items(meta_path: Path, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    items = [
        {
            "id": str(item["id"]),
            "question": str(item.get("question") or ""),
            "choices": list(item.get("choices") or []),
        }
        for item in load_jsonl(meta_path)
        if item.get("id")
    ]
    if limit and limit > 0:
        items = items[: int(limit)]
    return items


def candidates_for_binary_filter(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if is_yes_no_question(item.get("choices") or [], item.get("question") or ""):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        out.append({"id": item["id"], "question": question})
    return out


def judgment_record(item: dict, *, raw_response: str, verdict: str | None) -> dict[str, Any]:
    return {
        "id": item["id"],
        "question": item.get("question") or "",
        "verdict": verdict,
        "binary": True if verdict == "Yes" else (False if verdict == "No" else None),
        "reason": extract_reason(raw_response, verdict),
        "raw_response": raw_response,
        "scoring": "binary_question_text",
    }


def write_filter_outputs(
    judgments_path: Path,
    *,
    binary_ids_path: Path,
    open_ended_ids_path: Path,
    summary_path: Path,
    judge_model_id: str,
    n_skipped_yes_no: int,
    n_candidates: int,
) -> dict[str, Any]:
    records = load_jsonl(judgments_path) if judgments_path.is_file() else []
    by_id: dict[str, dict] = {}
    for item in records:
        qid = str(item.get("id") or "")
        if qid:
            by_id[qid] = item
    ordered = list(by_id.values())
    binary_ids = [str(item["id"]) for item in ordered if item.get("verdict") == "Yes"]
    open_ids = [str(item["id"]) for item in ordered if item.get("verdict") == "No"]
    n_unparsed = sum(1 for item in ordered if item.get("verdict") not in {"Yes", "No"})
    write_id_csv(binary_ids_path, binary_ids)
    write_id_csv(open_ended_ids_path, open_ids)
    summary = {
        "judge_model_id": judge_model_id,
        "scoring": "binary_question_text",
        "n_skipped_yes_no": n_skipped_yes_no,
        "n_candidates": n_candidates,
        "n_judged": len(ordered),
        "n_binary": len(binary_ids),
        "n_open_ended": len(open_ids),
        "n_unparsed": n_unparsed,
        "judgments_path": str(judgments_path),
        "binary_ids_path": str(binary_ids_path),
        "open_ended_ids_path": str(open_ended_ids_path),
    }
    write_json(summary_path, summary)
    return summary


async def classify_with_gemini(
    items: list[dict[str, Any]],
    judgments_path: Path,
    *,
    model_id: str,
    max_workers: int,
    qps: float,
    timeout: float,
    retries: int,
    retry_interval: float,
    max_output_tokens: int,
    thinking_level: str,
) -> dict[str, Any]:
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    removed = prune_incomplete_judgments(judgments_path)
    if removed:
        print(f"[gemini] pruned {removed} incomplete rows")
    completed = load_completed_ids(judgments_path)
    pending = [item for item in items if item["id"] not in completed]
    print(
        f"[gemini] judge={model_id} pending={len(pending)} "
        f"completed={len(completed)} -> {judgments_path}"
    )
    if not pending:
        return {"status": "already_done", "n_ok": 0, "n_fail": 0, "n_pending": 0}

    classifier = GeminiTextClassifier(
        model_name=model_id,
        system_instruction=BINARY_SYSTEM_PROMPT,
        api_max_retries=retries,
        api_retry_interval=retry_interval,
        qps=qps,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
    )
    semaphore = asyncio.Semaphore(max_workers)
    write_lock = asyncio.Lock()
    n_ok = 0
    n_fail = 0
    n_unparsed = 0

    async def _one(item: dict) -> None:
        nonlocal n_ok, n_fail, n_unparsed
        user_prompt = create_binary_user_prompt(item.get("question") or "")
        try:
            async with semaphore:
                raw = await classifier.call(item["id"], user_prompt)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"[gemini] failed {item['id']}: {exc}")
            return
        verdict = parse_yes_no_verdict(raw)
        if verdict is None:
            n_unparsed += 1
            print(f"[gemini] unparsed verdict for {item['id']}: {raw[:180]!r}")
        async with write_lock:
            write_jsonl(
                judgments_path,
                [judgment_record(item, raw_response=raw, verdict=verdict)],
                mode="a",
            )
            n_ok += 1
            if n_ok % 10 == 0 or n_ok == len(pending):
                print(f"[gemini] scored {n_ok}/{len(pending)}")

    await asyncio.gather(*[_one(item) for item in pending])
    return {
        "status": "ok",
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_unparsed": n_unparsed,
        "n_pending": len(pending),
    }


async def async_main(args: argparse.Namespace) -> dict:
    meta_path = Path(args.meta).expanduser().resolve()
    if not meta_path.is_file():
        raise SystemExit(
            f"MMAR-meta.jsonl not found at {meta_path}. "
            "Run seed_volume.py or point --meta at a local copy."
        )
    items = load_mmar_items(meta_path, limit=args.limit)
    candidates = candidates_for_binary_filter(items)
    n_skipped = len(items) - len(candidates)
    print(
        f"[filter] {len(items)} questions, {n_skipped} already yes/no, "
        f"{len(candidates)} sent to Gemini (question text only)"
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / JUDGMENTS_PATH.name
    binary_ids_path = out_dir / GEMINI_BINARY_IDS_PATH.name
    open_ended_ids_path = out_dir / OPEN_ENDED_IDS_PATH.name
    summary_path = out_dir / SUMMARY_PATH.name

    grade = await classify_with_gemini(
        candidates,
        judgments_path,
        model_id=args.gemini_model,
        max_workers=args.max_workers,
        qps=args.qps,
        timeout=args.timeout,
        retries=args.retries,
        retry_interval=args.retry_interval,
        max_output_tokens=args.max_tokens,
        thinking_level=args.thinking_level,
    )
    summary = write_filter_outputs(
        judgments_path,
        binary_ids_path=binary_ids_path,
        open_ended_ids_path=open_ended_ids_path,
        summary_path=summary_path,
        judge_model_id=args.gemini_model,
        n_skipped_yes_no=n_skipped,
        n_candidates=len(candidates),
    )
    result = {**grade, "summary": summary}
    print("[gemini] done:", result)
    print(f"[filter] binary IDs:     {binary_ids_path} ({summary['n_binary']})")
    print(f"[filter] open-ended IDs: {open_ended_ids_path} ({summary['n_open_ended']})")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta", default=str(DEFAULT_META))
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max MMAR questions before yes/no skip (0 = all).",
    )
    p.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    p.add_argument("--out-dir", default=str(PACKAGE_DIR))
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--retries", type=int, default=20)
    p.add_argument("--retry-interval", type=float, default=1.0)
    p.add_argument("--qps", type=float, default=4.0)
    p.add_argument("--max-workers", "-j", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--thinking-level",
        default="medium",
        help="Gemini 3 thinking_level: low, medium, or high.",
    )
    args = p.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
