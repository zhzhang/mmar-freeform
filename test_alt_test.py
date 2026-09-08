import unittest

from alt_test import compare_binary_judges, score_binary_judge


def _triple(index: int, pred: bool) -> tuple[str, dict[str, bool], bool]:
    ratings = {"alice": True, "bob": True, "carol": True}
    return (f"q{index}\tm\t0", ratings, pred)


class ScoreBinaryJudgeTests(unittest.TestCase):
    def test_preserves_stable_annotator_ids_across_varying_teams(self) -> None:
        people = ("alice", "bob", "carol", "dave")
        instances = []
        for index in range(40):
            present = [
                person
                for person_index, person in enumerate(people)
                if person_index != index % len(people)
            ]
            ratings = {
                person: (index + rating_index) % 2 == 0
                for rating_index, person in enumerate(reversed(present))
            }
            instances.append((f"item-{index}", ratings, index % 2 == 0))

        result = score_binary_judge(instances)

        self.assertEqual(result["n_annotators"], 4)
        self.assertEqual(set(result["per_annotator"]), set(people))
        self.assertEqual(
            {details["n"] for details in result["per_annotator"].values()},
            {30},
        )

    def test_supports_legacy_positional_ratings(self) -> None:
        instances = [
            (f"item-{index}", [True, False, True], True)
            for index in range(30)
        ]

        result = score_binary_judge(instances)

        self.assertEqual(result["n_annotators"], 3)
        self.assertEqual(set(result["per_annotator"]), {"0", "1", "2"})


class CompareBinaryJudgesTests(unittest.TestCase):
    def test_identical_judges_are_not_worse(self) -> None:
        instances = [_triple(index, True) for index in range(40)]
        result = compare_binary_judges(instances, instances)
        self.assertAlmostEqual(result["delta"], 0.0)
        self.assertGreater(result["p_a_worse"], 0.1)
        self.assertFalse(result["a_significantly_worse"])
        self.assertEqual(result["n"], 40)
        self.assertEqual(result["n_only_a"], 0)
        self.assertEqual(result["n_only_b"], 0)

    def test_always_wrong_is_significantly_worse(self) -> None:
        instances_a = [_triple(index, False) for index in range(40)]
        instances_b = [_triple(index, True) for index in range(40)]
        result = compare_binary_judges(instances_a, instances_b)
        self.assertLess(result["delta"], -0.5)
        self.assertTrue(result["a_significantly_worse"])
        self.assertLess(result["p_a_worse"], 0.05)
        self.assertIsNotNone(result["delta_hi"])
        self.assertLess(result["delta_hi"], 0.0)

    def test_uses_instance_intersection(self) -> None:
        common = [_triple(index, True) for index in range(40)]
        extra_a = [
            (f"extra{index}\tm\t0", {"alice": True, "bob": True, "carol": True}, False)
            for index in range(40)
        ]
        result = compare_binary_judges(common + extra_a, common)
        self.assertEqual(result["n"], 40)
        self.assertEqual(result["n_only_a"], 40)
        self.assertAlmostEqual(result["delta"], 0.0)
        self.assertFalse(result["a_significantly_worse"])


if __name__ == "__main__":
    unittest.main()
