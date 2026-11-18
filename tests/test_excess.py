"""Controlled tests for the exact random-order reference and excess penalty."""
import argparse
import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import unittest

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--starter',type=Path,default=HERE.parent/'course_starter')
args=parser.parse_args()
sys.path.append(str(args.starter.resolve()))
from nsc import layer_capacities,original_nsc,resource_estimate
spec=importlib.util.spec_from_file_location('v2_test_score',HERE.parent/'src/excess_score.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)


class ExcessTests(unittest.TestCase):
    """Check the permutation reference and the excess penalty."""
    def test_exact_permutation_expectation(self):
        """Verify exact permutation expectation."""
        for values in ([1,4],[1,2,5],[1,1,3,3],[0,1,2,4,7]):
            observed=statistics.mean(score.capacity_statistics(perm)['adjacent']
                                     for perm in itertools.permutations(values))
            self.assertAlmostEqual(observed,score.excess_statistics(values)['random_order_reference'],places=12)

    def test_hand_computed_orderings(self):
        """Verify hand computed orderings."""
        grouped=score.excess_statistics([1,1,3,3])
        alternating=score.excess_statistics([1,3,1,3])
        self.assertAlmostEqual(grouped['global_cv2'],.25)
        self.assertAlmostEqual(grouped['random_order_reference'],2/3)
        self.assertEqual(grouped['excess'],0)
        self.assertAlmostEqual(alternating['excess'],1/3)
        self.assertEqual(score.excess_capacity([1,1,3,3],1),8)
        self.assertAlmostEqual(score.excess_capacity([1,3,1,3],1),6)

    def test_lambda_zero_and_uniform(self):
        """Verify lambda zero and uniform."""
        cfg=dict(d_model=128,n_layer=3,n_head=4,d_inner=[512,1024,777])
        self.assertEqual(score.excess_score(cfg,0),original_nsc(cfg))
        for values in ([0,0,0],[3,3,3],[5]):
            self.assertEqual(score.excess_capacity(values,2),math.fsum(values))

    def test_single_and_two_layer_exact_zero_excess(self):
        """Verify single and two layer exact zero excess."""
        for values in ([7],[0],[1,17],[0,9],[1e-300,1e300]):
            self.assertEqual(score.excess_statistics(values)['excess'],0)
            self.assertEqual(score.excess_capacity(values,1e308),math.fsum(values))

    def test_reversal_invariance(self):
        """Verify reversal invariance."""
        for values in ([1,7,2,3],[0,8,0,8,1],[1,1,3,3]):
            self.assertEqual(score.excess_capacity(values,2),score.excess_capacity(values[::-1],2))

    def test_actual_reorder_and_reference_ablation(self):
        """Verify actual reorder and reference ablation."""
        a=dict(d_model=128,n_layer=4,n_head=[4]*4,d_inner=[512,512,2048,2048])
        b=copy.deepcopy(a);b['d_inner']=[512,2048,512,2048]
        self.assertEqual(original_nsc(a),original_nsc(b))
        self.assertEqual(resource_estimate(a),resource_estimate(b))
        self.assertGreater(score.excess_score(a,1),score.excess_score(b,1))
        self.assertEqual(score.global_score(a,1),score.global_score(b,1))
        self.assertEqual(score.excess_score(a,1),original_nsc(a))
        self.assertLess(score.adjacent_score(a,1),original_nsc(a))
        self.assertEqual(score.ablation(a),score.adjacent_score(a,score.SELECTED_LAMBDA))

    def test_scaling_and_large_coefficient(self):
        """Verify scaling and large coefficient."""
        values=[1,4,1,4]
        reference=score.excess_capacity(values,1)
        for scale in (1e-300,1e300):
            self.assertAlmostEqual(score.excess_capacity([v*scale for v in values],1)/scale,reference)
        result=score.excess_capacity([0,0,1,0],1e308)
        self.assertTrue(math.isfinite(result));self.assertGreater(result,0)
        self.assertTrue(math.isfinite(score.excess_statistics([0,0,5e-324])['excess']))

    def test_invalid_values_and_architectures(self):
        """Verify invalid values and architectures."""
        for values in ([],[-1,2],[float('nan')],[float('inf')],[True],['1'],[1e308,1e308]):
            with self.subTest(values=values),self.assertRaises(ValueError):score.excess_capacity(values,1)
        for coefficient in (-1,float('nan'),float('inf'),True,'1'):
            with self.subTest(coefficient=coefficient),self.assertRaises(ValueError):score.excess_capacity([1,2,3],coefficient)
        with self.assertRaises(ValueError):
            score.excess_score(dict(d_model=128,n_layer=0,n_head=4,d_inner=512),1)

    def test_all_development_and_input_preserved(self):
        """Verify all development and input preserved."""
        read=lambda name:json.loads((args.starter/'data'/name).read_text())
        configs=read('architectures.json');keys=read('splits.json')['development']
        for key in keys:
            cfg=configs[key];original=copy.deepcopy(cfg);base=original_nsc(cfg)
            self.assertEqual(score.excess_score(cfg,0),base)
            stats=score.excess_statistics(layer_capacities(cfg))
            self.assertGreaterEqual(stats['excess'],0)
            self.assertLessEqual(stats['excess'],stats['adjacent'])
            last=base
            for coefficient in (0,.1,.25,.5,1,2):
                value=score.excess_score(cfg,coefficient)
                self.assertTrue(math.isfinite(value));self.assertGreater(value,0)
                self.assertLessEqual(value,last);last=value
            self.assertEqual(cfg,original)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExcessTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
