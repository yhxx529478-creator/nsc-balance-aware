"""Controlled scoring and numerical tests on development configurations."""
import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--starter', type=Path,
                    default=Path(__file__).resolve().parent.parent / 'course_starter')
ARGS = PARSER.parse_args()
STARTER = ARGS.starter.resolve()
sys.path.append(str(STARTER))
from nsc import layer_capacities, original_nsc, resource_estimate
import importlib.util
SPEC = importlib.util.spec_from_file_location('tested_adjacent', Path(__file__).resolve().parents[1]/'src/adjacent_score.py')
student = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(student)


def config(inner, heads=4):
    """Build a small controlled architecture with the requested FFN widths."""
    return dict(d_model=128, n_layer=len(inner), n_head=heads, d_inner=list(inner))


class BalanceTests(unittest.TestCase):
    """Check adjacent and global penalties and their numerical boundaries."""
    def test_uniform_capacity(self):
        """Verify uniform capacity."""
        for coefficient in (0, .1, 1, 2, 1e308):
            self.assertEqual(student.penalized_capacity([3, 3, 3], coefficient), 9)
            self.assertEqual(student.penalized_capacity([3, 3, 3], coefficient, 'global_cv2'), 9)

    def test_hand_computed_mismatch(self):
        """Verify hand computed mismatch."""
        stats = student.capacity_statistics([1, 2, 3])
        self.assertAlmostEqual(stats['adjacent'], .25)
        self.assertAlmostEqual(stats['global_cv2'], 1 / 6)
        self.assertAlmostEqual(student.penalized_capacity([1, 2, 3], 2), 4)

    def test_equal_total_different_order(self):
        """Verify equal total different order."""
        grouped, alternating = [1, 1, 3, 3], [1, 3, 1, 3]
        self.assertEqual(sum(grouped), sum(alternating))
        self.assertAlmostEqual(student.capacity_statistics(alternating)['adjacent'],
                               3 * student.capacity_statistics(grouped)['adjacent'])
        self.assertGreater(student.penalized_capacity(grouped, 1),
                           student.penalized_capacity(alternating, 1))
        self.assertEqual(student.penalized_capacity(grouped, 1, 'global_cv2'),
                         student.penalized_capacity(alternating, 1, 'global_cv2'))

    def test_actual_architecture_reordering(self):
        """Verify actual architecture reordering."""
        grouped = config([512, 512, 2048, 2048])
        alternating = config([512, 2048, 512, 2048])
        self.assertEqual(original_nsc(grouped), original_nsc(alternating))
        self.assertEqual(resource_estimate(grouped), resource_estimate(alternating))
        self.assertGreater(student.adjacent_score(grouped, 1), student.adjacent_score(alternating, 1))
        self.assertEqual(student.global_score(grouped, 1), student.global_score(alternating, 1))

    def test_reversal_invariance_including_two_layers(self):
        """Verify reversal invariance including two layers."""
        for values in ([1, 5], [1, 7, 2, 3], [2, 2, 9, 1, 4]):
            self.assertAlmostEqual(student.penalized_capacity(values, 2),
                                   student.penalized_capacity(values[::-1], 2))

    def test_global_permutation_invariance(self):
        """Verify global permutation invariance."""
        values = [1, 2, 3, 5]
        reference = student.penalized_capacity(values, .5, 'global_cv2')
        for perm in itertools.permutations(values):
            self.assertEqual(student.penalized_capacity(perm, .5, 'global_cv2'), reference)

    def test_strong_imbalance_and_monotone_penalty(self):
        """Verify strong imbalance and monotone penalty."""
        mild, strong = [1, 1, 2, 2], [0, 0, 6, 0]
        self.assertGreater(student.capacity_statistics(strong)['adjacent'],
                           student.capacity_statistics(mild)['adjacent'])
        self.assertLess(student.penalized_capacity(strong, 1), student.penalized_capacity(mild, 1))
        scores = [student.penalized_capacity(strong, x) for x in (0, .1, .5, 1, 2)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_lambda_zero_exact_baseline(self):
        """Verify lambda zero exact baseline."""
        cfg = config([512, 1024, 777])
        self.assertEqual(student.adjacent_score(cfg, 0), original_nsc(cfg))
        self.assertEqual(student.global_score(cfg, 0), original_nsc(cfg))
        with patch.object(student, 'SELECTED_LAMBDA', 0):
            self.assertEqual(student.score(cfg), original_nsc(cfg))
            self.assertEqual(student.ablation(cfg), original_nsc(cfg))

    def test_single_layer(self):
        """Verify single layer."""
        for values in ([0], [3], [1e-300]):
            self.assertEqual(student.capacity_statistics(values)['adjacent'], 0)
            self.assertEqual(student.capacity_statistics(values)['global_cv2'], 0)
            self.assertEqual(student.penalized_capacity(values, 2), values[0])
        cfg = config([512])
        self.assertEqual(student.adjacent_score(cfg, 2), original_nsc(cfg))

    def test_zero_mean(self):
        """Verify zero mean."""
        for mechanism in ('adjacent', 'global_cv2'):
            self.assertEqual(student.penalized_capacity([0, 0, 0], 2, mechanism), 0)

    def test_scale_invariance_and_large_small_values(self):
        """Verify scale invariance and large small values."""
        base = [1, 2, 3]
        for scale in (1e-300, 1e300):
            scaled = [x * scale for x in base]
            stats = student.capacity_statistics(scaled)
            self.assertAlmostEqual(stats['adjacent'], .25)
            self.assertAlmostEqual(student.penalized_capacity(scaled, 1) / scale,
                                   student.penalized_capacity(base, 1))
        self.assertAlmostEqual(student.capacity_statistics([0, 0, 5e-324])['adjacent'], 4.5)
        result = student.penalized_capacity([0, 0, 1], 1e308)
        self.assertGreater(result, 0)
        self.assertTrue(math.isfinite(result))

    def test_invalid_capacities_and_coefficient(self):
        """Verify invalid capacities and coefficient."""
        for values in ([], [-1, 2], [float('nan')], [float('inf')], [True], ['2'], [1e308, 1e308]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                student.penalized_capacity(values, 1)
        for coefficient in (-1, float('nan'), float('inf'), True, '1'):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                student.adjacent_score(config([512]), coefficient)
        with self.assertRaises(ValueError):
            student.penalized_capacity([1], 1, 'invalid')

    def test_invalid_architecture(self):
        """Verify invalid architecture."""
        for cfg in (dict(d_model=128, n_layer=0, n_head=4, d_inner=512),
                    dict(d_model=128, n_layer=2, n_head=[4], d_inner=[512, 512]),
                    dict(d_model=128, n_layer=1, n_head=3, d_inner=512)):
            with self.assertRaises(ValueError):
                student.adjacent_score(cfg, 1)

    def test_development_configurations_and_no_mutation(self):
        """Verify development configurations and no mutation."""
        configs = json.loads((STARTER / 'data' / 'architectures.json').read_text())
        keys = json.loads((STARTER / 'data' / 'splits.json').read_text())['development']
        for key in keys:
            cfg = configs[key]
            before = copy.deepcopy(cfg)
            total = original_nsc(cfg)
            self.assertEqual(student.adjacent_score(cfg, 0), total)
            for coefficient in (0, .1, .25, .5, 1, 2):
                for function in (student.adjacent_score, student.global_score):
                    value = function(cfg, coefficient)
                    self.assertTrue(math.isfinite(value))
                    self.assertGreater(value, 0)
                    self.assertLessEqual(value, total)
            self.assertEqual(cfg, before)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BalanceTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not outcome.wasSuccessful())
