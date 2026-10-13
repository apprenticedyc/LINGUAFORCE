"""Guard against scoring thinking text or silently repairing invalid labels."""
import unittest

import numpy as np
from sklearn.metrics import cohen_kappa_score

from score_qwen4b import parse_answer, valid, metrics


class QwenScoringTests(unittest.TestCase):
    def test_final_answer_only(self):
        answer = '<think>{"binary":0}</think>\n```json\n{"binary":1}\n```'
        self.assertEqual(parse_answer(answer), {"binary": 1})

    def test_ambiguous_answers_rejected(self):
        for answer in ('<think>{"binary":1}', '{"binary":1,"binary":0}',
                       '{"binary":1} extra', '[{"binary":1}]'):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                parse_answer(answer)

    def test_task_validity_is_separate(self):
        answer = parse_answer('{"binary":1,"intensity":6,"dimensions":{"D1":0}}')
        self.assertTrue(valid("pressure", answer))
        self.assertFalse(valid("intensity", answer))
        self.assertFalse(valid("dimensions", answer))
        self.assertFalse(valid("pressure", {"binary": True}))

    def test_dimension_mean_and_fixed_scale(self):
        y = np.tile(np.array([0, 1, 2, 3])[:, None], (1, 7))
        p = y.copy()
        p[-1, :] = 0
        result = metrics("dimensions", y, p)
        reference = cohen_kappa_score(y[:, 0], p[:, 0], labels=[0, 1, 2, 3], weights="quadratic")
        self.assertAlmostEqual(result["qwk"], reference)
        self.assertAlmostEqual(result["mae"], 0.75)


if __name__ == "__main__":
    unittest.main()
