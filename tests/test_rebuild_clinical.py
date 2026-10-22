import tempfile
from pathlib import Path
import unittest
import numpy as np
from preprocessing.rebuild_clinical import (assert_disjoint, canonical_patient,
    global_tuab_split, parse_summary_strict, window_plan)

class ClinicalRebuildTests(unittest.TestCase):
    def test_eval_grid_never_depends_on_label(self):
        a,y,_=window_plan(105,10,10,[(9,11),(30,60)])
        b,zero,_=window_plan(105,10,10,[])
        np.testing.assert_array_equal(a,b)
        self.assertEqual(y.tolist(),[1,1,0,1,1,1,0,0,0,0])
        self.assertEqual(zero.sum(),0)
    def test_half_open_and_containment(self):
        _, y, _ = window_plan(30,10,10,[(10,20)])
        self.assertEqual(y.tolist(),[0,1,0])
        _, y, _ = window_plan(30,10,10,[(0,30)])
        self.assertEqual(y.tolist(),[1,1,1])
    def test_augmentation_truth_not_force_positive(self):
        _, y, aug=window_plan(50,10,10,[(20,21)],
            {'pre_samples':20,'post_samples':20,'stride_samples':5})
        self.assertTrue(aug.any())
        self.assertTrue((y[aug]==0).any())
    def test_chb_alias_prevents_leakage(self):
        with self.assertRaises(ValueError):
            assert_disjoint({'train':[canonical_patient('chb01')],
                             'val':[canonical_patient('chb21')],'test':['chb24']})
    def test_tuab_both_diagnoses_stay_together(self):
        rows=[{'patient':f'p{k}','label':c} for k in range(10) for c in (0,1)]
        grouped=global_tuab_split(rows,[{'patient':'p10','label':0}])
        self.assertEqual(len(grouped['train']),16)
        self.assertEqual(len(grouped['val']),4)
        with self.assertRaises(ValueError):
            global_tuab_split(rows,[{'patient':'p1','label':0}])
    def test_malformed_annotation_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'summary'
            p.write_text('File Name: x.edf\nNumber of Seizures in File: 1\nSeizure Start Time: 2 seconds\n')
            with self.assertRaises(ValueError):parse_summary_strict(p)
            p.write_text('File Name: x.edf\nNumber of Seizures in File: 1\nSeizure 1 Start Time: 2 seconds\nSeizure 1 End Time: 4 seconds\n')
            self.assertEqual(parse_summary_strict(p),{'x.edf':[(2,4)]})
if __name__=='__main__':unittest.main()
