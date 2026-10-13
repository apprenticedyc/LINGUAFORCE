"""Check bootstrap statistics against independent scoring implementations."""
import unittest
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
from analyze_human_robustness import control_groups, ordinal_metrics, strategy_metrics, wilson


class RobustnessTests(unittest.TestCase):
    def test_ordinal_resampling_matches_sklearn(self):
        y = np.array([[0, 0], [1, 3], [2, 1], [3, 2]])
        p = np.array([[1, 0], [0, 2], [2, 3], [1, 2]])
        indices = [0, 0, 1, 3]
        weights = np.array([np.bincount(indices, minlength=4)])
        result = ordinal_metrics(y, p, 3, weights)
        self.assertAlmostEqual(result['mae'][0], mean_absolute_error(y[indices], p[indices]))
        expected = np.mean([cohen_kappa_score(y[indices, d], p[indices, d],
            labels=range(4), weights='quadratic') for d in range(2)])
        self.assertAlmostEqual(result['qwk'][0], expected)

    def test_strategy_resampling_keeps_all_labels(self):
        y = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        p = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        indices = [0, 0, 2]
        weights = np.array([np.bincount(indices, minlength=3)])
        result = strategy_metrics(y, p, weights)
        for average in ('macro', 'micro'):
            self.assertAlmostEqual(result[average + '_f1'][0],
                f1_score(y[indices], p[indices], average=average, zero_division=0))

    def test_undefined_qwk_is_not_zero(self):
        result = ordinal_metrics([0, 0], [0, 0], 3, np.ones((1, 2)))
        self.assertTrue(np.isnan(result['qwk'][0]))

    def test_controls_use_human_labels_and_deduplicate(self):
        def row(binary, request, reason):
            return {'binary': binary, 'strategies': {'A1_Request_Advice': request,
                    'A3_Rational_Persuasion': reason}}
        groups = control_groups({'a': row(0, 1, 1), 'b': row(1, 1, 0), 'c': row(0, 0, 0)})
        self.assertEqual(groups['request_or_reason'], ['a'])
        self.assertEqual(groups['all_negative'], ['a', 'c'])

    def test_zero_errors_still_have_interval(self):
        low, high = wilson(0, 126)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0)
        self.assertLess(high, 0.04)


if __name__ == '__main__':
    unittest.main()
