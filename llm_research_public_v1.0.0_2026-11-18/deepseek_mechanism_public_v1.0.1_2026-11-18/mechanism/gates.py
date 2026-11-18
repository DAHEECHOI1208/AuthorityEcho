"""Mechanical implementation gates, distinct from scientific efficacy claims."""
from pathlib import Path
from .io import require,read,atomic,load_run,result_id,committed_record,file_hash
from .interventions import load_cohort,plans
from .judging import verdict


def baseline_gate(run):
    run=Path(run);m,rows=load_run(run);fail=[]
    for r in rows:
        tid=result_id(r);a=committed_record(run,tid);v=verdict(run,tid)
        if a is None or a.get('technical_status')!='ok':fail.append({'sample_id':r['sample_id'],'reason':'missing_or_failed_target'})
        if v is None or v.get('judgment_status')!='ok':fail.append({'sample_id':r['sample_id'],'reason':'missing_or_failed_judge'})
    result={'passed':not fail,'failures':fail,'run_identity':m['run_identity'],'response_validity_separate_from_completion':True}
    atomic(run/'baseline_gate.json',result);require(result['passed'],'Baseline mechanical gate failed; inspect baseline_gate.json');return result


def intervention_gate(run,smoke=True):
    run=Path(run);m,rows=load_run(run);c=load_cohort(run);p=plans(rows,c,m['study']);p=p[:max(6,len(p)//max(1,len(c.get('generation_members',c['members']))))] if smoke else p
    checks=[];fail=[];warnings=[]
    for r,arm,l,a in p:
        tid=result_id(r,arm,l,a);out=committed_record(run,tid);base=committed_record(run,result_id(r));v=verdict(run,tid)
        if not out or out.get('technical_status')!='ok':fail.append({'task':tid,'reason':'target_missing_or_failed'});continue
        if not v or v.get('judgment_status')!='ok':fail.append({'task':tid,'reason':'judge_missing_or_failed'})
        effects=out.get('diagnostics',{}).get('effects',[])
        if len(effects)!=1:fail.append({'task':tid,'reason':'wrong_hook_count'});continue
        e=effects[0];okay=e['layer']==l and e['position']=='t_A' and e['phase']=='prefill' and e['applications']==1 and e['decode_edits']==0 and e['other_tokens_bitwise_equal']
        if not okay:fail.append({'task':tid,'reason':'wrong_hook_location'})
        if arm=='sham' and (e['actual_delta_norm']!=0 or out['response_token_ids']!=base['response_token_ids']):fail.append({'task':tid,'reason':'sham_token_or_vector_mismatch'})
        if out.get('truncated') is not False or out.get('stop_reason')!='eos':warnings.append({'task':tid,'reason':'incomplete_output_excluded_from_strict_metrics'})
        if e.get('pre_edit_drift_l2',0)>0:warnings.append({'task':tid,'reason':'pre_edit_numerical_drift','l2':e['pre_edit_drift_l2']})
        if e.get('relative_norm_error',0)>.02:warnings.append({'task':tid,'reason':'random_BF16_norm_mismatch','relative_error':e['relative_norm_error']})
        checks.append({'task':tid,'effect':e})
    result={'passed':not fail,'checks':checks,'failures':fail,'warnings':warnings,'fresh_RA_n':len(c['members']),
      'empty_cohort':not c['members'],'efficacy_claim':False,'source_files':{f'outputs/{x["task"]}.json':file_hash(run/'outputs'/f'{x["task"]}.json') for x in checks}}
    atomic(run/('intervention_smoke.json' if smoke else 'intervention_gate.json'),result)
    require(result['passed'],'Intervention mechanical gate failed');return {'passed':True,'checked':len(checks),'warnings':len(warnings)}
