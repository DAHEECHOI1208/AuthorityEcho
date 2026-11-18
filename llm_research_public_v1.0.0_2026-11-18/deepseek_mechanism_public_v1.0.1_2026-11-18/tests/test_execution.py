import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import urllib.error
from unittest.mock import patch
import numpy as np
from mechanism.geometry_base import bf16
from mechanism.hooks import Capture
from mechanism.io import ROOT,read,atomic,load_run,result_id,file_hash
from mechanism.inputs import prepare
from mechanism import judge_transport as transport
from mechanism.judging import evaluate
import test_contracts

class Tensor:
    """Small deterministic CPU double; never represented as a real torch test."""
    dtype='torch.bfloat16';device='synthetic_cpu';is_cuda=False
    def __init__(self,x):self.x=np.asarray(x)
    @property
    def shape(self):return self.x.shape
    @property
    def ndim(self):return self.x.ndim
    def __getitem__(self,k):return Tensor(self.x[k])
    def __setitem__(self,k,v):self.x[k]=v.x if isinstance(v,Tensor) else v
    def detach(self):return self
    def float(self):return Tensor(self.x.astype(np.float32))
    def cpu(self):return self
    def numpy(self):return self.x
    def clone(self):return Tensor(self.x.copy())
    def mean(self,dim=0):return Tensor(self.x.mean(axis=dim))
    def tolist(self):return self.x.tolist()
    def to(self,**kwargs):return Tensor(bf16(self.x))

class Handle:
    def __init__(self,seq,f):self.seq=seq;self.f=f
    def remove(self):self.seq.remove(self.f)

class Block:
    def __init__(self):self.hooks=[]
    def register_forward_hook(self,f):self.hooks.append(f);return Handle(self.hooks,f)

class Model:
    def __init__(self):self.pre=[];self.config=types.SimpleNamespace(hidden_size=2048);self.layers=[Block() for _ in range(27)]
    def register_forward_pre_hook(self,f,with_kwargs=True):self.pre.append(f);return Handle(self.pre,f)
    def call(self,ids,cache=None):
        for f in self.pre:f(self,(),{'input_ids':Tensor([ids]),'past_key_values':cache})
        out=Tensor(np.ones((1,len(ids),2048),np.float32))
        for block in self.layers:
            for f in block.hooks:out=f(block,(),out)
        return out

class HookTests(unittest.TestCase):
    def setUp(self):
        self.model=Model();self.adapter=types.SimpleNamespace(model=self.model,layers=self.model.layers)
        self.serial={'input_ids':[1,2,3,4],'positions':{'t_U':[1],'t_D':[2],'t_C':[2],'t_A':[3]}}
        self.torch=types.SimpleNamespace(from_numpy=lambda a:Tensor(a),equal=lambda a,b:np.array_equal(a.x,b.x))
    def test_exact_block_token_once_and_decode_unchanged(self):
        donor=np.full(2048,2,np.float32);patchspec={'layer':16,'arm':'full_replace','donor':donor,'expected_before':np.ones(2048)}
        with patch.dict(sys.modules,{'torch':self.torch}):
            with Capture(self.adapter,self.serial,patchspec) as trace:
                pre=self.model.call(self.serial['input_ids']);dec=self.model.call([9],types.SimpleNamespace(get_seq_length=lambda:4))
        np.testing.assert_array_equal(pre.x[0,:3],1);np.testing.assert_array_equal(pre.x[0,3],donor);np.testing.assert_array_equal(dec.x,1)
        self.assertEqual(trace.applications,1);self.assertEqual(trace.effects[0]['layer'],16);self.assertEqual(trace.effects[0]['decode_edits'],0);self.assertFalse(self.model.pre);self.assertTrue(all(not b.hooks for b in self.model.layers))
    def test_independent_layer20_and_sham(self):
        patchspec={'layer':20,'arm':'sham','donor':np.ones(2048)}
        with patch.dict(sys.modules,{'torch':self.torch}):
            with Capture(self.adapter,self.serial,patchspec) as trace:self.model.call(self.serial['input_ids'])
        self.assertEqual(trace.applications,1);self.assertEqual(trace.effects[0]['actual_delta_norm'],0)
    def test_nonempty_initial_cache_rejected(self):
        with self.assertRaises(ValueError):
            with Capture(self.adapter,self.serial) as trace:self.model.call(self.serial['input_ids'],types.SimpleNamespace(get_seq_length=lambda:2))
        self.assertFalse(self.model.pre)
    def test_repeated_prefill_not_treated_as_decode(self):
        with self.assertRaises(ValueError):
            with Capture(self.adapter,self.serial) as trace:self.model.call(self.serial['input_ids']);self.model.call(self.serial['input_ids'])
    def test_bad_full_vector_before_state_rejected(self):
        with patch.dict(sys.modules,{'torch':self.torch}):
            with self.assertRaises(ValueError):
                with Capture(self.adapter,self.serial,{'layer':16,'arm':'full_replace','donor':np.ones(2048),'expected_before':np.zeros(2048)}) as trace:self.model.call(self.serial['input_ids'])
    def test_missing_block_rejected(self):
        with self.assertRaises(ValueError):
            with Capture(self.adapter,self.serial) as trace:
                trace.root(self.model,(),{'input_ids':Tensor([self.serial['input_ids']])});trace.block(0,Tensor(np.ones((1,4,2048),np.float32)))

class JudgeTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name);self.cfg=read(ROOT/'configs/judge.json');self.cfg['api_key_env']='SYNTHETIC_TEST_TOKEN'
        self.old={k:getattr(transport,k) for k in ('RUBRIC_VERSION','RUBRIC','LABELS','SCHEMA','REASON_LANGUAGE_PATTERN')}
        transport.RUBRIC_VERSION='test_only';transport.RUBRIC='Synthetic test rubric';transport.LABELS=['a','b'];transport.SCHEMA={};transport.REASON_LANGUAGE_PATTERN='[a-z]'
        self.env=patch.dict('os.environ',{'SYNTHETIC_TEST_TOKEN':'not-a-real-key-for-unit-test'});self.env.start()
    def tearDown(self):
        for k,v in self.old.items():setattr(transport,k,v)
        self.env.stop();self.temp.cleanup()
    def test_auth_failure_not_retried_or_replayed(self):
        def fail(*args,**kwargs):raise urllib.error.HTTPError('https://example.invalid',401,'Unauthorized',{},io.BytesIO(b'{}'))
        with patch.object(transport.urllib.request,'urlopen',side_effect=fail) as mock:
            r=transport.judge_one({'question':'Synthetic','response_text':'Synthetic'},self.cfg,self.path/'cache',sleep=lambda _:None)
            r2=transport.judge_one({'question':'Synthetic','response_text':'Synthetic'},self.cfg,self.path/'cache',sleep=lambda _:None)
        self.assertEqual(mock.call_count,1);self.assertEqual(r,r2);self.assertEqual(r['failure_cause'],'authentication_or_permission');self.assertEqual(r['attempt_count'],1)
        self.assertNotIn('not-a-real-key-for-unit-test',''.join(p.read_text() for p in self.path.rglob('*.json')))
    def test_blinded_payload_ignores_arm(self):
        a=transport.payload_for({'question':'q','response_text':'r','arm':'restore','layer':16},self.cfg);b=transport.payload_for({'question':'q','response_text':'r','arm':'random','layer':20},self.cfg);self.assertEqual(a,b)
    def test_provider_length_is_failure_not_refusal(self):
        r=transport.parse_response({'choices':[{'finish_reason':'length','message':{'content':'{"label":"a","reason":"x"}'}}]});self.assertEqual(r['failure_cause'],'judge_truncated')
    def test_duplicate_json_label_is_failure(self):
        r=transport.parse_response({'choices':[{'finish_reason':'stop','message':{'content':'{"label":"a","label":"b","reason":"x"}'}}]});self.assertEqual(r['failure_cause'],'invalid_json')
    def test_orchestrator_stops_after_first_failed_item(self):
        run=self.path/'run';prepare(ROOT/'input.json',run);m,rows=load_run(run)
        for row in rows[:2]:atomic(run/'outputs'/f'{result_id(row)}.json',{**row,'task_id':result_id(row),'run_identity':m['run_identity'],'arm':'baseline','layer':None,'alpha':None,'technical_status':'ok','response_text':'Synthetic'})
        with patch('mechanism.judging.configure',return_value={}),patch.object(transport,'judge_one',return_value={'judgment_status':'failed','failure_cause':'authentication_or_permission'}) as mock:
            with self.assertRaises(RuntimeError):evaluate(run)
        self.assertEqual(mock.call_count,1);self.assertEqual(len(list((run/'judges/main/links').glob('*.json'))),1)
    def test_wrong_run_rejected_before_judge_call(self):
        run=self.path/'run';prepare(ROOT/'input.json',run);m,rows=load_run(run);row=rows[0]
        atomic(run/'outputs'/f'{result_id(row)}.json',{**row,'task_id':result_id(row),'run_identity':'other_run','arm':'baseline','technical_status':'ok','response_text':'Synthetic'})
        with patch('mechanism.judging.configure',return_value={}),patch.object(transport,'judge_one') as mock:
            with self.assertRaises(ValueError):evaluate(run)
        mock.assert_not_called()

class RuntimeResumeTests(unittest.TestCase):
    def test_completed_output_skips_generation(self):
        from mechanism.runtime import generate_one
        helper=test_contracts.SyntheticRunTests();helper.setUp()
        try:
            adapter=types.SimpleNamespace(serialize=lambda row:(_ for _ in ()).throw(AssertionError('Must not generate')))
            r=generate_one(adapter,helper.run,helper.rows[0],helper.m['run_identity'],'synthetic_only');self.assertEqual(r['technical_status'],'ok')
        finally:helper.tearDown()

if __name__=='__main__':unittest.main()
