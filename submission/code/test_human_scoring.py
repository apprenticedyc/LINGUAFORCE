"""Focused checks for ordinal scoring and malformed-output handling."""
import unittest
from score_human_llm import qwk, decode, ordinal, score
from run_human_llm import STRATEGIES
import json


class ScoringTests(unittest.TestCase):
    def test_quadratic_weights(self):
        # One three-level error has squared distance 9; expected disagreement is 10.
        self.assertAlmostEqual(qwk([0,1,2,3],[0,1,2,0],3),0.1)

    def test_alias_preserves_value(self):
        output=decode({'text':json.dumps({'strategies':{'D3_Logical_False_Dilemma':1}})})
        self.assertEqual(output['strategies'],{'D3_Logical_Trap_False_Dilemma':1})

    def test_ordinal_rejects_out_of_range_and_boolean(self):
        self.assertFalse(ordinal(4,3))
        self.assertFalse(ordinal(True,3))
        self.assertTrue(ordinal(3,3))

    def test_missing_outer_brace_only(self):
        self.assertEqual(decode({'text':'{"binary":1,"dimensions":{"D1":2}'}),
                         {'binary':1,'dimensions':{'D1':2}})
        self.assertEqual(decode({'text':'{"binary":1,"dimensions":{"D1":'}),{})

    def test_invalid_dimension_does_not_drop_other_tasks(self):
        gold={str(i):{'binary':i,'intensity':i,'dimensions':{f'D{d}':i for d in range(1,8)},
                     'strategies':{s:i for s in STRATEGIES}} for i in (0,1)}
        records=[]
        for i,g in gold.items():
            p=json.loads(json.dumps(g));p['pressure_score']=float(g['binary'])
            if i=='1':p['dimensions']['D1']=4
            records.append({'id':i,'text':json.dumps(p)})
        result=score(records,gold)
        self.assertEqual(result['pressure']['n'],2)
        self.assertEqual(result['pressure']['auroc'],1)
        self.assertEqual(result['dimension_profile_n'],1)
        self.assertAlmostEqual(result['dimension_exact_accuracy_all'],13/14)


if __name__=='__main__':unittest.main()
