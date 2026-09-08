"""Alternative Annotator Test (Calderon et al., 2025).

Port of the official procedure in https://github.com/nitaytech/AltTest
(``alt_test_example.ipynb``). Average advantage probability ρ is the
primary score for comparing judges. Winning rate ω depends on ε and is
secondary.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import ttest_1samp

DEFAULT_EPSILON = 0.15
DEFAULT_Q_FDR = 0.05
MIN_HUMANS_PER_INSTANCE = 3
MIN_INSTANCES_PER_HUMAN = 30
PASS_THRESHOLD = 0.5
# Sentinel for a stored judge reply with no pass/fail verdict. It never equals a
# boolean human rating, so accuracy against remaining annotators is 0.
UNPARSED: Any = "__unparsed__"
DEFAULT_CI_LEVEL = 0.95
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_BOOTSTRAP_SEED = 0

InstanceId = Hashable
AnnotatorId = Hashable
Annotation = Any
ScoringFn = Callable[[Any, list[Any]], float]
HumanRatings = Mapping[AnnotatorId, bool] | Sequence[bool]


def majority_label(ratings: Sequence[bool]) -> bool | None:
    """Strict majority of boolean ratings, or None on a tie / empty list."""
    if not ratings:
        return None
    n_true = sum(1 for value in ratings if value)
    n_false = len(ratings) - n_true
    if n_true == n_false:
        return None
    return n_true > n_false


def scoring_gold(
    ratings: Sequence[bool], *, min_humans: int = MIN_HUMANS_PER_INSTANCE
) -> bool | None:
    """Gold used for per-shot diagnostics: majority when ≥ ``min_humans`` ratings."""
    values = [bool(item) for item in ratings]
    if len(values) < min_humans:
        return bool(values[0]) if values else None
    majority = majority_label(values)
    if majority is not None:
        return majority
    return bool(values[0])


def accuracy(pred: Any, annotations: Sequence[Any]) -> float:
    if not annotations:
        return 0.0
    return float(np.mean([pred == ann for ann in annotations]))


def llm_annotation_from_entry(entry: object) -> Any:
    """Judge label for the Alt-Test, or ``None`` if the slot is missing.

    Parsed ``pass``/``fail`` become booleans. Unparsed text is ``UNPARSED``, which
    does not match either human rating (disagreement, not an Incorrect verdict).
    """
    if not isinstance(entry, dict):
        return None
    verdict = str(entry.get("verdict") or "").strip().lower()
    if verdict not in {"pass", "fail"}:
        return UNPARSED
    if entry.get("correct") is not None:
        return bool(entry.get("correct"))
    return verdict == "pass"


def by_procedure(p_values: Sequence[float], q: float) -> list[int]:
    """Benjamini–Yekutieli FDR: indices of rejected nulls (original order)."""
    if not p_values:
        return []
    p_arr = np.array(p_values, dtype=float)
    m = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_pvals = p_arr[sorted_indices]
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    thresholds = (np.arange(1, m + 1) / m) * (q / harmonic)
    max_i = -1
    for i in range(m):
        if sorted_pvals[i] <= thresholds[i]:
            max_i = i
    if max_i == -1:
        return []
    return [int(idx) for idx in sorted_indices[: max_i + 1]]


def ttest(indicators: Sequence[float], epsilon: float) -> float:
    arr = np.asarray(indicators, dtype=float)
    if len(arr) < 2:
        return 1.0
    if np.allclose(arr, arr[0]):
        return 0.0 if float(arr[0]) < epsilon else 1.0
    pvalue = ttest_1samp(arr, epsilon, alternative="less").pvalue
    if pvalue is None or not np.isfinite(pvalue):
        return 1.0
    return float(pvalue)


def _instance_cluster_key(instance_id: InstanceId) -> str:
    """Question id when ``instance_id`` is ``question\\tmodel\\tshot``, else the id."""
    text = str(instance_id)
    if "\t" in text:
        return text.split("\t", 1)[0]
    return text


def _cluster_table(
    per_annotator_items: Mapping[str, Sequence[tuple[str, float]]],
) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    """Annotator → cluster → W^f values, plus the sorted cluster id list."""
    cluster_vals: dict[str, dict[str, list[float]]] = {}
    for annotator, items in per_annotator_items.items():
        by_cluster: dict[str, list[float]] = {}
        for cluster, value in items:
            by_cluster.setdefault(str(cluster), []).append(float(value))
        if by_cluster:
            cluster_vals[str(annotator)] = by_cluster
    clusters = sorted({c for by_c in cluster_vals.values() for c in by_c})
    return cluster_vals, clusters


def _rho_on_draw(
    cluster_vals: Mapping[str, Mapping[str, Sequence[float]]],
    clusters: Sequence[str],
    chosen: Sequence[int],
) -> float:
    rhos: list[float] = []
    for by_cluster in cluster_vals.values():
        vals: list[float] = []
        for index in chosen:
            vals.extend(by_cluster.get(clusters[int(index)], ()))
        if vals:
            rhos.append(float(np.mean(vals)))
    return float(np.mean(rhos)) if rhos else float("nan")


def _rho_from_items(
    per_annotator_items: Mapping[str, Sequence[tuple[str, float]]],
) -> float | None:
    rhos: list[float] = []
    for rows in per_annotator_items.values():
        if rows:
            rhos.append(float(np.mean([float(value) for _cluster, value in rows])))
    return float(np.mean(rhos)) if rhos else None


def _bootstrap_rho_ci(
    per_annotator_items: Mapping[str, Sequence[tuple[str, float]]],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float] | tuple[None, None]:
    """Percentile CI for ρ by resampling question clusters with replacement.

    ``per_annotator_items`` maps annotator id → ``(cluster_key, W^f)`` rows.
    """
    cluster_vals, clusters = _cluster_table(per_annotator_items)
    if len(clusters) < 2 or n_bootstrap < 1:
        return None, None

    rng = np.random.default_rng(seed)
    n_clusters = len(clusters)
    alpha = (1.0 - float(ci_level)) / 2.0
    draws = np.empty(int(n_bootstrap), dtype=float)
    for b in range(int(n_bootstrap)):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        draws[b] = _rho_on_draw(cluster_vals, clusters, chosen)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return None, None
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _bootstrap_rho_delta(
    items_a: Mapping[str, Sequence[tuple[str, float]]],
    items_b: Mapping[str, Sequence[tuple[str, float]]],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[np.ndarray, float | None]:
    """Paired cluster draws of Δ = ρ_a − ρ_b. Returns (finite draws, one-sided upper)."""
    table_a, clusters_a = _cluster_table(items_a)
    table_b, clusters_b = _cluster_table(items_b)
    shared_annotators = sorted(set(table_a) & set(table_b))
    table_a = {key: table_a[key] for key in shared_annotators}
    table_b = {key: table_b[key] for key in shared_annotators}
    clusters = sorted(set(clusters_a) | set(clusters_b))
    if len(clusters) < 2 or n_bootstrap < 1 or not shared_annotators:
        return np.array([], dtype=float), None

    rng = np.random.default_rng(seed)
    n_clusters = len(clusters)
    draws = np.empty(int(n_bootstrap), dtype=float)
    for b in range(int(n_bootstrap)):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        draws[b] = _rho_on_draw(table_a, clusters, chosen) - _rho_on_draw(
            table_b, clusters, chosen
        )
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return finite, None
    return finite, float(np.quantile(finite, float(ci_level)))


def _resolve_scoring_function(scoring_function: str | ScoringFn) -> ScoringFn:
    if isinstance(scoring_function, str):
        if scoring_function == "accuracy":
            return accuracy
        raise ValueError(f"Unknown scoring function: {scoring_function}")
    return scoring_function


def leave_one_out_alignments(
    llm_annotations: Mapping[InstanceId, Annotation],
    humans_annotations: Mapping[AnnotatorId, Mapping[InstanceId, Annotation]],
    scoring_function: str | ScoringFn = "accuracy",
    min_humans_per_instance: int = MIN_HUMANS_PER_INSTANCE,
    min_instances_per_human: int = MIN_INSTANCES_PER_HUMAN,
) -> tuple[dict[str, list[tuple[InstanceId, float, float]]], list[str], int]:
    """Per excluded annotator, ``(instance_id, S_llm, S_human)`` vs remaining labelers.

    ``S`` is the Alt-Test alignment score (ACC for discrete labels): agreement
    with the annotators who were not left out. Advantage indicators are
    ``S_llm >= S_human`` / ``S_human >= S_llm``.
    """
    score_fn = _resolve_scoring_function(scoring_function)

    i_set: dict[AnnotatorId, list[InstanceId]] = {}
    h_set: dict[InstanceId, list[AnnotatorId]] = {}
    for human, anns in humans_annotations.items():
        i_set[human] = list(anns.keys())
        for instance_id, _ann in anns.items():
            h_set.setdefault(instance_id, []).append(human)

    instances_to_keep = {
        instance_id
        for instance_id, annotators in h_set.items()
        if len(annotators) >= min_humans_per_instance and instance_id in llm_annotations
    }
    i_set = {
        human: [instance_id for instance_id in ids if instance_id in instances_to_keep]
        for human, ids in i_set.items()
    }
    h_set = {
        instance_id: annotators
        for instance_id, annotators in h_set.items()
        if instance_id in instances_to_keep
    }

    alignments: dict[str, list[tuple[InstanceId, float, float]]] = {}
    skipped_annotators: list[str] = []
    for excluded_h in humans_annotations:
        instances = [i for i in i_set.get(excluded_h, []) if i in llm_annotations]
        if len(instances) < min_instances_per_human:
            skipped_annotators.append(str(excluded_h))
            continue
        pairs: list[tuple[InstanceId, float, float]] = []
        for instance_id in instances:
            human_ann = humans_annotations[excluded_h][instance_id]
            llm_ann = llm_annotations[instance_id]
            remaining = [
                humans_annotations[other][instance_id]
                for other in h_set[instance_id]
                if other != excluded_h
            ]
            pairs.append(
                (
                    instance_id,
                    score_fn(llm_ann, remaining),
                    score_fn(human_ann, remaining),
                )
            )
        alignments[str(excluded_h)] = pairs
    return alignments, skipped_annotators, len(instances_to_keep)


def alt_test(
    llm_annotations: Mapping[InstanceId, Annotation],
    humans_annotations: Mapping[AnnotatorId, Mapping[InstanceId, Annotation]],
    scoring_function: str | ScoringFn = "accuracy",
    epsilon: float = DEFAULT_EPSILON,
    q_fdr: float = DEFAULT_Q_FDR,
    min_humans_per_instance: int = MIN_HUMANS_PER_INSTANCE,
    min_instances_per_human: int = MIN_INSTANCES_PER_HUMAN,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Leave-one-out Alt-Test. ρ does not depend on ``epsilon``; ω does.

    ``advantage_prob_ci_low`` / ``_high`` are a ``ci_level`` percentile interval
    from resampling question clusters (see ``_instance_cluster_key``).
    """
    alignments, skipped_annotators, n_kept = leave_one_out_alignments(
        llm_annotations,
        humans_annotations,
        scoring_function=scoring_function,
        min_humans_per_instance=min_humans_per_instance,
        min_instances_per_human=min_instances_per_human,
    )

    p_values: list[float] = []
    advantage_probs: list[float] = []
    agree_llm: list[float] = []
    agree_human: list[float] = []
    humans: list[str] = []
    per_annotator: dict[str, dict[str, Any]] = {}
    bootstrap_items: dict[str, list[tuple[str, float]]] = {}

    for excluded_h, pairs in alignments.items():
        llm_scores = [llm_score for _iid, llm_score, _human in pairs]
        human_scores = [human_score for _iid, _llm, human_score in pairs]
        llm_indicators = [
            1 if llm_score >= human_score else 0
            for _iid, llm_score, human_score in pairs
        ]
        excluded_indicators = [
            1 if human_score >= llm_score else 0
            for _iid, llm_score, human_score in pairs
        ]
        diffs = [
            float(exc_ind - llm_ind)
            for exc_ind, llm_ind in zip(excluded_indicators, llm_indicators)
        ]
        rho_f = float(np.mean(llm_indicators)) if llm_indicators else 0.0
        rho_h = float(np.mean(excluded_indicators)) if excluded_indicators else 0.0
        alignment_f = float(np.mean(llm_scores)) if llm_scores else 0.0
        alignment_h = float(np.mean(human_scores)) if human_scores else 0.0
        pvalue = ttest(diffs, epsilon)
        p_values.append(pvalue)
        advantage_probs.append(rho_f)
        agree_llm.append(alignment_f)
        agree_human.append(alignment_h)
        humans.append(excluded_h)
        per_annotator[excluded_h] = {
            "rho_f": rho_f,
            "rho_h": rho_h,
            "alignment_f": alignment_f,
            "alignment_h": alignment_h,
            "p_value": pvalue,
            "n": len(pairs),
            "rejected": False,
        }
        bootstrap_items[excluded_h] = [
            (
                _instance_cluster_key(instance_id),
                float(llm_ind),
            )
            for (instance_id, _s_llm, _s_hum), llm_ind in zip(pairs, llm_indicators)
        ]

    rejected = set(by_procedure(p_values, q_fdr))
    for index in rejected:
        key = humans[index]
        if key in per_annotator:
            per_annotator[key]["rejected"] = True

    n_humans = len(humans)
    winning_rate = (len(rejected) / n_humans) if n_humans else None
    advantage_prob = float(np.mean(advantage_probs)) if advantage_probs else None
    ci_low, ci_high = _bootstrap_rho_ci(
        bootstrap_items,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=bootstrap_seed,
    )
    return {
        "advantage_prob": advantage_prob,
        "advantage_prob_ci_low": ci_low,
        "advantage_prob_ci_high": ci_high,
        "advantage_prob_ci_level": float(ci_level),
        "winning_rate": winning_rate,
        "passed": (winning_rate >= PASS_THRESHOLD)
        if winning_rate is not None
        else None,
        "loo_agree_judge": float(np.mean(agree_llm)) if agree_llm else None,
        "loo_agree_human": float(np.mean(agree_human)) if agree_human else None,
        "n_annotators": n_humans,
        "epsilon": float(epsilon),
        "n": n_kept,
        "per_annotator": per_annotator,
        "skipped_annotators": skipped_annotators,
    }


def _maps_from_instances(
    instances: Sequence[tuple[InstanceId, HumanRatings, Any]],
    *,
    min_humans_per_instance: int = MIN_HUMANS_PER_INSTANCE,
) -> tuple[dict[InstanceId, Any], dict[AnnotatorId, dict[InstanceId, bool]], int, int]:
    """LLM labels and per-annotator maps. Drops ``None`` preds and thin rating sets."""
    humans: dict[AnnotatorId, dict[InstanceId, bool]] = {}
    llm: dict[InstanceId, Any] = {}
    n_missing = 0
    n_skipped_lt3 = 0
    for instance_id, ratings, pred in instances:
        if isinstance(ratings, Mapping):
            annotated_values = [
                (annotator_id, bool(rating)) for annotator_id, rating in ratings.items()
            ]
        else:
            annotated_values = [
                (str(index), bool(rating)) for index, rating in enumerate(ratings)
            ]
        if len(annotated_values) < min_humans_per_instance:
            n_skipped_lt3 += 1
            continue
        if pred is None:
            n_missing += 1
            continue
        llm[instance_id] = pred
        for annotator_id, rating in annotated_values:
            humans.setdefault(annotator_id, {})[instance_id] = rating
    return llm, humans, n_missing, n_skipped_lt3


def _advantage_items(
    llm: Mapping[InstanceId, Annotation],
    humans: Mapping[AnnotatorId, Mapping[InstanceId, Annotation]],
    *,
    min_humans_per_instance: int = MIN_HUMANS_PER_INSTANCE,
    min_instances_per_human: int = MIN_INSTANCES_PER_HUMAN,
) -> dict[str, list[tuple[str, float]]]:
    """Per excluded annotator, ``(cluster_key, W^f)`` rows."""
    alignments, _skipped, _n_kept = leave_one_out_alignments(
        llm,
        humans,
        scoring_function="accuracy",
        min_humans_per_instance=min_humans_per_instance,
        min_instances_per_human=min_instances_per_human,
    )
    items: dict[str, list[tuple[str, float]]] = {}
    for excluded_h, pairs in alignments.items():
        items[str(excluded_h)] = [
            (
                _instance_cluster_key(instance_id),
                1.0 if llm_score >= human_score else 0.0,
            )
            for instance_id, llm_score, human_score in pairs
        ]
    return items


def score_binary_judge(
    instances: Sequence[tuple[InstanceId, HumanRatings, Any]],
    *,
    epsilon: float = DEFAULT_EPSILON,
    min_humans_per_instance: int = MIN_HUMANS_PER_INSTANCE,
    min_instances_per_human: int = MIN_INSTANCES_PER_HUMAN,
    q_fdr: float = DEFAULT_Q_FDR,
) -> dict[str, Any]:
    """Score one judge against boolean human ratings.

    Each item is ``(instance_id, ratings, llm_verdict)``. ``ratings`` should be
    an annotator-id-to-label mapping so the same human is excluded across all
    of their instances. Legacy boolean sequences remain supported and use the
    rating index as the annotator id.

    ``llm_verdict`` is ``None`` when the judge slot is missing, a bool for a
    parsed pass/fail, or ``UNPARSED`` when the stored text has no verdict.
    Unparsed replies enter the test and never match a human bool (disagreement).
    Only instances with at least ``min_humans_per_instance`` ratings and a
    non-``None`` label enter.
    """
    n_total = len(instances)
    llm, humans, n_missing, n_skipped_lt3 = _maps_from_instances(
        instances, min_humans_per_instance=min_humans_per_instance
    )

    if not llm or not humans:
        return {
            "advantage_prob": None,
            "advantage_prob_ci_low": None,
            "advantage_prob_ci_high": None,
            "advantage_prob_ci_level": float(DEFAULT_CI_LEVEL),
            "winning_rate": None,
            "passed": None,
            "loo_agree_judge": None,
            "loo_agree_human": None,
            "n": 0,
            "n_missing": n_missing,
            "n_skipped_lt3": n_skipped_lt3,
            "n_label_rows": n_total,
            "n_annotators": 0,
            "epsilon": float(epsilon),
            "per_annotator": {},
            "skipped_annotators": [],
        }

    result = alt_test(
        llm,
        humans,
        scoring_function="accuracy",
        epsilon=epsilon,
        q_fdr=q_fdr,
        min_humans_per_instance=min_humans_per_instance,
        min_instances_per_human=min_instances_per_human,
    )
    result["n"] = len(llm)
    result["n_missing"] = n_missing
    result["n_skipped_lt3"] = n_skipped_lt3
    result["n_label_rows"] = n_total
    return result


def compare_binary_judges(
    instances_a: Sequence[tuple[InstanceId, HumanRatings, Any]],
    instances_b: Sequence[tuple[InstanceId, HumanRatings, Any]],
    *,
    min_humans_per_instance: int = MIN_HUMANS_PER_INSTANCE,
    min_instances_per_human: int = MIN_INSTANCES_PER_HUMAN,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired cluster bootstrap of Δ = ρ_a − ρ_b.

    Instance ids must have a non-``None`` label on both sides; ratings come
    from ``instances_a`` on that intersection. The one-sided test is
    H1: ρ_a < ρ_b (A is worse). ``p_a_worse`` is plus-one corrected
    P*(Δ* ≥ 0). ``a_significantly_worse`` is true when the one-sided
    ``ci_level`` upper bound on Δ is below 0.
    """
    llm_a, humans_a, n_missing_a, n_skipped_a = _maps_from_instances(
        instances_a, min_humans_per_instance=min_humans_per_instance
    )
    llm_b, _humans_b, n_missing_b, n_skipped_b = _maps_from_instances(
        instances_b, min_humans_per_instance=min_humans_per_instance
    )
    ids_a, ids_b = set(llm_a), set(llm_b)
    common = ids_a & ids_b
    n_only_a = len(ids_a - common)
    n_only_b = len(ids_b - common)
    llm_a = {key: llm_a[key] for key in common}
    llm_b = {key: llm_b[key] for key in common}
    humans = {
        annotator: {iid: label for iid, label in ratings.items() if iid in common}
        for annotator, ratings in humans_a.items()
    }
    humans = {annotator: ratings for annotator, ratings in humans.items() if ratings}

    empty = {
        "rho_a": None,
        "rho_b": None,
        "delta": None,
        "delta_hi": None,
        "p_a_worse": None,
        "a_significantly_worse": False,
        "n": 0,
        "n_only_a": n_only_a,
        "n_only_b": n_only_b,
        "n_missing_a": n_missing_a,
        "n_missing_b": n_missing_b,
        "n_skipped_lt3_a": n_skipped_a,
        "n_skipped_lt3_b": n_skipped_b,
        "n_annotators": 0,
        "ci_level": float(ci_level),
        "n_bootstrap": 0,
    }
    if not common or not humans:
        return empty

    items_a = _advantage_items(
        llm_a,
        humans,
        min_humans_per_instance=min_humans_per_instance,
        min_instances_per_human=min_instances_per_human,
    )
    items_b = _advantage_items(
        llm_b,
        humans,
        min_humans_per_instance=min_humans_per_instance,
        min_instances_per_human=min_instances_per_human,
    )
    shared = sorted(set(items_a) & set(items_b))
    items_a = {key: items_a[key] for key in shared}
    items_b = {key: items_b[key] for key in shared}
    rho_a = _rho_from_items(items_a)
    rho_b = _rho_from_items(items_b)
    if rho_a is None or rho_b is None:
        empty["n"] = len(common)
        empty["n_annotators"] = len(shared)
        return empty

    draws, delta_hi = _bootstrap_rho_delta(
        items_a,
        items_b,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=bootstrap_seed,
    )
    n_finite = int(draws.size)
    if n_finite:
        p_a_worse = (1.0 + float(np.sum(draws >= 0.0))) / (n_finite + 1.0)
    else:
        p_a_worse = None
    delta = float(rho_a - rho_b)
    worse = bool(delta_hi is not None and delta_hi < 0.0)
    return {
        "rho_a": float(rho_a),
        "rho_b": float(rho_b),
        "delta": delta,
        "delta_hi": delta_hi,
        "p_a_worse": p_a_worse,
        "a_significantly_worse": worse,
        "n": len(common),
        "n_only_a": n_only_a,
        "n_only_b": n_only_b,
        "n_missing_a": n_missing_a,
        "n_missing_b": n_missing_b,
        "n_skipped_lt3_a": n_skipped_a,
        "n_skipped_lt3_b": n_skipped_b,
        "n_annotators": len(shared),
        "ci_level": float(ci_level),
        "n_bootstrap": n_finite,
    }
