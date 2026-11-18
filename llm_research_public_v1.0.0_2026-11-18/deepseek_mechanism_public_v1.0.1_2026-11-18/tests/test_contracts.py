import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from mechanism.io import ROOT,read,atomic,file_hash,npz_save,result_id,load_run,committed_record,digest,jsonl
from mechanism.inputs import rows_for_case,load_cases,prepare,cross_rows
from mechanism.native import P19_deepseek as ds
from mechanism.geometry import edit
from mechanism.geometry_base import bf16,direction,restore
from mechanism.statistics_base import paired
from mechanism.analysis import paired_report,analyze
from mechanism.judging import metric,status
from mechanism.interventions import freeze,load_cohort,plans

class CharacterTokenizer:
    bos_token='<B>';eos_token='<E>';bos_token_id=1;chat_template=ds.NATIVE_TEMPLATE
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        text=self.bos_token
        for m in messages:
            if m['role']=='system':text+=m['content']+'\n\n'
            elif m['role']=='user':text+='User: '+m['content']+'\n\n'
            elif m['role']=='assistant':text+='Assistant: '+m['content']+self.eos_token
        return text+('Assistant:' if add_generation_prompt else '')
    def __call__(self,text,**kwargs):
        # Deterministic test tokenizer, not a claim about real DeepSeek IDs.
        return {'input_ids':[1]+[ord(c)+10 for c in text[3:]],'offset_mapping':[(0,3)]+[(i,i+1) for i in range(3,len(text))]}

class InputTests(unittest.TestCase):
    def setUp(self):self.case=read(ROOT/'input.json');self.rows=rows_for_case(self.case)
    def test_exact_six_conditions(self):self.assertEqual([r['condition_id'] for r in self.rows],['C0','C1','C2','C3-E','C3-P','C4'])
    def test_persona_document_assignment(self):
        r={x['condition_id']:x for x in self.rows}
        self.assertEqual(r['C0']['persona'],'');self.assertEqual(r['C1']['document'],'');self.assertEqual(r['C2']['document'],r['C4']['document']);self.assertEqual(r['C2']['persona'],'')
    def test_native_serialization_and_all_positions(self):
        for r in self.rows:
            s=ds.serialize(r,read(ROOT/'configs/target.json'),CharacterTokenizer());self.assertIn('Tool (retrieve_reference):',s['text']);self.assertEqual(s['positions']['t_A'],[len(s['input_ids'])-1]);self.assertEqual(bool(s['positions']['t_D']),bool(r['document']))
    def test_json_order_preserved_across_disk(self):
        r=self.rows[-1];before=ds.render(r,CharacterTokenizer())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'row.json';atomic(p,r);self.assertEqual(before,ds.render(read(p),CharacterTokenizer()))
    def test_lossy_native_tool_template_rejected(self):
        r=copy.deepcopy(self.rows[-1]);r['messages'][2]['tool_calls'][0]['id']='bad'
        with self.assertRaises(Exception):ds.render(r,CharacterTokenizer())
    def test_changed_chat_template_rejected(self):
        t=CharacterTokenizer();t.chat_template='changed'
        with self.assertRaises(Exception):ds.render(self.rows[-1],t)
    def test_input_limit_does_not_truncate(self):
        cfg=read(ROOT/'configs/target.json');cfg['max_input_tokens']=4
        with self.assertRaises(Exception):ds.serialize(self.rows[-1],cfg,CharacterTokenizer())
    def test_new_question_allowed_without_old_hash(self):
        self.case['question']='A new synthetic test question.'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'case.json';atomic(p,self.case);prepare(p,Path(d)/'run');m,rows=load_run(Path(d)/'run');self.assertEqual(rows[0]['question'],self.case['question'])
    def test_duplicate_case_rejected(self):
        m=dict(set_id='s',variant='harmful',parent_question_id='p',shared_component_id='g',dev_role='readout_check')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'corpus.json';atomic(p,{'cases':[{'input':self.case,'metadata':m}]*2})
            with self.assertRaises(ValueError):load_cases(p,True)
    def test_component_leakage_rejected(self):
        a={'input':self.case,'metadata':dict(set_id='s',variant='harmful',parent_question_id='p',shared_component_id='g',dev_role='calibration_fit')};b=copy.deepcopy(a);b['input']['question_id']='another';b['metadata']['dev_role']='readout_check'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'corpus.json';atomic(p,{'cases':[a,b]})
            with self.assertRaises(ValueError):load_cases(p,True)
    def test_cross_eight_conditions(self):
        config={'templates':{'single:harmful:example_001':{'entity_B':'A different fictional entity role.','purpose_B':'A different fictional purpose role.'}}}
        r=cross_rows(self.rows,config);self.assertEqual(len(r),8);self.assertEqual(len({x['sample_id'] for x in r}),8)
        for x in r:self.assertEqual(x['question'],self.case['question'])

class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(29);self.h=bf16(rng.normal(size=2048));self.d=bf16(rng.normal(size=2048));self.mu=rng.normal(size=(3,2048))
    def test_zero_alpha_identity(self):
        y,r,a=edit(self.h,self.d,'restore',0.,self.mu);np.testing.assert_array_equal(y,self.h);self.assertEqual(r['actual_delta_norm'],0)
    def test_original_alpha_one_exact(self):
        y,r,a=edit(self.h,self.d,'restore',1.,self.mu);u=direction(self.mu);expected=bf16(self.h.astype(float)+((self.d.astype(float)-self.h)@u)*u);np.testing.assert_array_equal(y,expected)
    def test_orthogonal_component_preserved_before_rounding(self):
        y,r,a=edit(self.h,self.d,'restore',1.5,self.mu);self.assertLess(r['intended_orthogonal_delta_norm'],1e-12);self.assertGreaterEqual(r['actual_orthogonal_delta_norm'],0)
    def test_full_vector_is_not_projection(self):
        full,info,_=edit(self.h,self.d,'full_replace',mu=self.mu);partial,_,_=edit(self.h,self.d,'restore',mu=self.mu);np.testing.assert_array_equal(full,bf16(self.d));self.assertFalse(np.array_equal(full,partial));self.assertTrue(info['full_donor_bf16_equal'])
    def test_random_deterministic_actual_norm_match(self):
        _,r,_=edit(self.h,self.d,'restore',1.5,self.mu)
        a,x,_=edit(self.h,self.d,'random',1.5,self.mu,identity='test',random_target=r['actual_delta_norm']);b,y,_=edit(self.h,self.d,'random',1.5,self.mu,identity='test',random_target=r['actual_delta_norm']);np.testing.assert_array_equal(a,b);self.assertLess(x['relative_norm_error'],.05)
    def test_shape_nonfinite_and_strength_rejected(self):
        for bad in (self.d[:-1],np.full(2048,np.nan)):
            with self.assertRaises(ValueError):edit(self.h,bad,'restore',mu=self.mu)
        with self.assertRaises(ValueError):edit(self.h,self.d,'restore',3.,self.mu)
    def test_hrmean_not_per_item_donor(self):
        a,_,_=edit(self.h,self.d,'hrmean',mu=self.mu);b,_,_=edit(self.h,-self.d,'hrmean',mu=self.mu);np.testing.assert_array_equal(a,b)

class StatisticsTests(unittest.TestCase):
    def test_two_intervals_and_unchanged_original(self):
        a=np.array([[1,0],[1,1],[1,0],[1,1],[1,np.nan],[1,0]]);g=['a','a','b','b','c','d'];old=paired(a,g,family=2);new=paired_report(a,g,2)
        self.assertEqual(new['ci_pp'],old['ci_pp']);self.assertEqual(new['difference_pp'],old['difference_pp']);self.assertEqual(new['ci95_pointwise_pp'],paired(a,g,family=1)['ci_pp']);self.assertEqual(new['reduction_ci95_pointwise_pp'],[-x for x in reversed(new['ci95_pointwise_pp'])])
    def test_missing_bounds_and_matching_denominators(self):
        r=paired([[1,0],[1,None]],['a','b']);self.assertEqual(r['denominator'],1);self.assertEqual(r['all_eligible_missingness_bounds_pp'],[-100.,-50.]);self.assertEqual(r['arm_missing_n'],1)
    def test_empty_and_single_cluster_have_no_CI(self):
        for a,g in [(np.empty((0,2)),[]),([[1,0],[1,1]],['a','a'])]:self.assertIsNone(paired_report(a,g)['ci95_pointwise_pp'])
    def test_weighted_auc_ties(self):
        from mechanism.validation_base import weighted_auc
        self.assertEqual(weighted_auc([1,1],[True,False],[2,3]),.5)
    def test_alert_threshold_conservative_ties(self):
        from mechanism.alert_base import calibrate_threshold
        x=np.array([0,1,1,1,2]);t=calibrate_threshold(x,.2);self.assertLessEqual((x>t).sum(),1)
    def test_joint_bootstrap_uses_shared_draw(self):
        from mechanism.joint_bootstrap import joint_bootstrap
        from mechanism import validation_base as v
        rs=[];X=[]
        for group,c,cl in __import__('itertools').product(('HR','HA','BR','BA'),v.MAIN,('a','b','c')):
            rs.append({'group':group,'condition_id':c,'shared_component_id':cl,'parent_question_id':group+c+cl});X.append([2 if group in ('HR','BR') else 1,1 if group=='BA' else .3])
        b,values,valid=joint_bootstrap(v,np,rs,np.array(X),['a','b','c'],{'bootstrap_repeats':20,'bootstrap_seed':20261023,'minimum_valid_bootstrap_fraction':.95});self.assertEqual(valid.sum(),20);self.assertTrue(all(x['CI_scope']=='apparent_same_sample_candidatewise_not_selection_adjusted' for x in b.values()))

class ValidityTests(unittest.TestCase):
    def setUp(self):self.row=dict(technical_status='ok',response_text='Synthetic response.',truncated=False,stop_reason='eos',output_kind='text_or_data');self.v=dict(judgment_status='ok',canonical_label='refusal')
    def test_failure_is_not_refusal(self):
        for row,v in [(None,self.v),(self.row,None),({**self.row,'truncated':True},self.v),(self.row,{**self.v,'canonical_label':'undetermined'}),(self.row,{**self.v,'judgment_status':'failed'})]:self.assertIsNone(metric(row,v,'refusal'))
    def test_partial_compliance_is_compliance(self):self.assertEqual(metric(self.row,{**self.v,'canonical_label':'partially_compliant'}),1)
    def test_clarification_distinct_metric_denominators(self):
        v={**self.v,'canonical_label':'clarification_request'};self.assertEqual(metric(self.row,v),0);self.assertIsNone(metric(self.row,v,'paper_hcr'));self.assertEqual(metric(self.row,v,'refusal'),0)

class SyntheticRunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.run=Path(self.temp.name)/'run';prepare(ROOT/'input.json',self.run);self.m,self.rows=load_run(self.run)
        self.arr=np.ones((27,4,2048),dtype=np.float32)
        for row in self.rows:self.record(row,'refusal' if row['condition_id']=='C0' else 'compliant')
    def tearDown(self):self.temp.cleanup()
    def record(self,row,label,arm='baseline',layer=None,alpha=None):
        tid=result_id(row,arm,layer,alpha);dest=self.run/'activations'/f'{tid}.npz';npz_save(dest,activations=self.arr,positions=np.asarray(['t_U','t_D','t_C','t_A']),layers=np.arange(27))
        r={**row,'task_id':tid,'run_identity':self.m['run_identity'],'runtime_hash':'synthetic_only','arm':arm,'layer':layer,'alpha':alpha,'activation_file':str(dest.relative_to(self.run)),'activation_sha256':file_hash(dest),'technical_status':'ok','response_text':'Synthetic test response.','response_token_ids':[2],'truncated':False,'stop_reason':'eos','output_kind':'text_or_data','serialized':{'input_ids':[1,2]}}
        p=self.run/'outputs'/f'{tid}.json';atomic(p,r);atomic(self.run/'judges/main/links'/f'{tid}.json',{'task_id':tid,'run_identity':self.m['run_identity'],'judgment_status':'ok','canonical_label':label,'response_record_sha256':file_hash(p)})
    def test_fresh_RA_requires_both_fresh_baselines(self):
        self.record(self.rows[0],'compliant');freeze(self.run);self.assertEqual(load_cohort(self.run)['members'],[])
    def test_partial_C4_is_fresh_RA(self):
        self.record(self.rows[-1],'partially_compliant');freeze(self.run);self.assertEqual(len(load_cohort(self.run)['members']),1)
    def test_failed_intervention_does_not_change_membership(self):
        freeze(self.run);c=load_cohort(self.run);atomic(self.run/'outputs/failure.json',{'task_id':'failure','technical_status':'failed'});self.assertEqual(c,load_cohort(self.run));self.assertEqual(len(plans(self.rows,c,'restore')),6)
    def test_frozen_baseline_mutation_rejected(self):
        freeze(self.run);self.record(self.rows[0],'compliant')
        with self.assertRaises(ValueError):load_cohort(self.run)
    def test_activation_tampering_rejected(self):
        tid=result_id(self.rows[0]);npz_save(self.run/'activations'/f'{tid}.npz',activations=self.arr+1)
        with self.assertRaises(ValueError):committed_record(self.run,tid)
    def test_saved_analysis_and_score_export(self):
        freeze(self.run);out=Path(self.temp.name)/'analysis';r=analyze(self.run,out);self.assertEqual(r['fresh_RA'],1);self.assertTrue((out/'result.json').exists())
        from mechanism.scoring import export_ledger,score_ledger
        ledger=Path(self.temp.name)/'ledger.jsonl';export_ledger(self.run,ledger);mu=np.ones((27,3,2048));mu[:,1,0]=2
        centers=Path(self.temp.name)/'centers.npz';npz_save(centers,centers=mu,groups=np.asarray(['HR','HA','BA']),layers=np.arange(27),position=np.asarray('t_A'));scores=Path(self.temp.name)/'scores.jsonl';self.assertEqual(score_ledger(ledger,centers,scores)['score_rows'],648)
    def test_readonly_status_preserves_files(self):
        from mechanism.operations import status
        before={str(p):file_hash(p) for p in self.run.rglob('*') if p.is_file()};status(self.run);self.assertEqual(before,{str(p):file_hash(p) for p in self.run.rglob('*') if p.is_file()})
    def test_wrong_task_id_rejected(self):
        tid=result_id(self.rows[0]);p=self.run/'outputs'/f'{tid}.json';r=read(p);r['task_id']='wrong';atomic(p,r)
        with self.assertRaises(ValueError):committed_record(self.run,tid)

if __name__=='__main__':unittest.main()
