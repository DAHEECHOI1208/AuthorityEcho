import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mechanism.io import atomic,jsonl,read,file_hash,read_jsonl
from mechanism.readouts import auc_curves,common_contrasts,cross_report,alert_analysis
from mechanism.regression import build_features,regression
from mechanism.probes import clustered
from mechanism.validation_base import centers


def synthetic_scores():
    rows=[]
    for i in range(6):
      for variant in ('harmful','benign'):
       for condition in ('C0','C3-E','C3-P','C4'):
        for layer in range(27):
            c={'C0':0,'C3-E':.02,'C3-P':.03,'C4':.05}[condition]
            g0=.01*(i%2)-.03;g=g0+c*(1 if variant=='harmful' else .2);f=.1-c
            group='BA' if variant=='benign' else 'HA' if condition!='C0' and i%2 else 'HR'
            rows.append({'task_id':f'{i}:{variant}:{condition}','sample_id':f'{i}:{variant}:{condition}',
             'set_id':'synthetic','variant':variant,'condition_id':condition,'parent_question_id':str(i),'question_id':variant+str(i),
             'layer_index':layer,'position':'t_A','dev_role':'readout_select' if i<2 else 'readout_check','arm':'baseline','group':group,
             'shared_component_id':'g'+str(i//2),'f':f,'g':g,'a':f+g,'cos_HR':.3,'cos_HA':.3+g,'cos_BA':.3-f,
             'input_tokens':12+int(c*100),'question_chars':10+i,'document_chars':50+int(c*100),'depth1':'test',
             'entity_s':c,'purpose_s':2*c})
    return rows

class AnalysisPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.rows=synthetic_scores();self.scores=self.root/'scores.jsonl';jsonl(self.scores,self.rows)
    def tearDown(self):self.tmp.cleanup()
    def test_additive_identity_common_same_parent(self):
        common_contrasts(self.scores,self.root/'common');rows=read_jsonl(self.root/'common/common_cohort_audit.jsonl');good=[r for r in rows if r['eligible']];self.assertTrue(good)
        self.assertLess(max(abs(r['entity_additivity_residual']) for r in good),1e-12)
    def test_duplicate_scores_rejected(self):
        jsonl(self.scores,self.rows+[self.rows[0]])
        with self.assertRaises(ValueError):common_contrasts(self.scores,self.root/'common')
    def test_fixed_alert_calibration_not_check(self):
        out=self.root/'alert.json';alert_analysis(self.scores,out);r=read(out)['results'];self.assertTrue(r);self.assertTrue(all(not x['threshold_recalibrated_on_evaluation'] for x in r));self.assertEqual(r[0]['calibration_negative_n'],1)
    def test_auc_missing_BR_remains_missing(self):
        auc_curves(self.scores,self.root/'aucs.csv');self.assertIn('missing_class_or_insufficient_components',(self.root/'aucs.csv').read_text())
    def test_regression_features_center_common_component(self):
        f,ex=build_features(self.scores);self.assertEqual(len(f),6*27);r=f[0];self.assertAlmostEqual(r['H_dm'],(r['H_dg']-r['H_df'])/3);self.assertIn('B_added_tokens',r)
    def test_regression_never_leaks_component_folds(self):
        # One held-out pair per component has both classes; select/check remains separate.
        r=regression(self.scores,self.root/'regression');self.assertGreater(r['models'],0)
        models=read(self.root/'regression/fitted_fold_models.json')
        for m in models:self.assertFalse(set(m['train_ids'])&set(m['test_ids']))
    def test_cross_difference_of_differences(self):
        r=[]
        for i in range(3):
            base=copy.deepcopy(self.rows[0]);base.update(parent_question_id=str(i),shared_component_id='g'+str(i))
            for cell,v in [('AA',4.),('AB',1.),('BA',2.),('BB',5.)]:r.append({**base,'condition_id':'entity_'+cell,'entity_s':v})
        jsonl(self.scores,r);cross_report(self.scores,self.root/'cross');ledger=read_jsonl(self.root/'cross/cross_pairs.jsonl');s=next(x for x in ledger if x['metric']=='entity_s');self.assertEqual(s['difference_in_differences'],6.)
    def test_probe_four_cluster_enumeration(self):
        r=clustered(np.array([1.,2.,3.,4.]),np.array([2.,3.,4.,5.]),['a','b','c','d']);self.assertEqual(r['draws'],256);self.assertTrue(r['enumerated'])
    def test_center_fit_balances_conditions_not_row_count(self):
        rows=[];x=[]
        for group in ('HR','BA'):
            for i,c in enumerate(('C2','C3-E','C3-P','C4')):
                for p in range(i+1):
                    rows.append({'group':group,'condition_id':c,'parent_question_id':group+c+str(p),'shared_component_id':'g'});x.append([i+1,1 if group=='HR' else 2])
        mu=centers(rows,np.asarray(x,float));np.testing.assert_allclose(mu,[[2.5,1],[2.5,2]])

if __name__=='__main__':unittest.main()

class AlignmentContractTests(unittest.TestCase):
    def test_same_shared_direction_is_not_entity_specific(self):
        from mechanism.directions import load_directions
        from mechanism.io import npz_save
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'directions.npz';a=np.zeros((2,27,4,2048));a[0,...,0]=1;a[1,...,1]=1
            npz_save(p,directions=a,contrasts=np.asarray(['C3-E','C3-P']),layers=np.arange(27),positions=np.asarray(['t_U','t_D','t_C','t_A']))
            direct,shared=load_directions(p);self.assertAlmostEqual(shared[12,3,0],1/np.sqrt(2));self.assertAlmostEqual(shared[12,3,1],1/np.sqrt(2));self.assertFalse(np.array_equal(shared[12,3],direct[0,12,3]))
    def test_original_alignment_cluster_rule(self):
        from mechanism.alignment_statistics import ClusterSample
        a=ClusterSample([{'cluster':str(i)} for i in range(4)]);self.assertEqual(len(a.weights)-1,256)
        b=ClusterSample([{'cluster':str(i)} for i in range(5)]);self.assertEqual(len(b.weights)-1,4000)
    def test_alignment_ratio_suppressed_for_near_zero_gap(self):
        from mechanism.alignment_statistics import ClusterSample
        s=ClusterSample([{'cluster':'a'},{'cluster':'b'}]);r=s.recovery([.1,.2],[0.,0.]);self.assertEqual(r['ratio_status'],'near_zero_gap');self.assertIsNone(r['point'])
