"""Single-case public input and explicit private corpus support."""
import copy
import json
import re
from pathlib import Path
from .io import ROOT,require,read,digest,file_hash,atomic,atomic_bytes,jsonl,code_hash,utc
from .native.P17_final_answer import construct_final

CONDITIONS=('C0','C1','C2','C3-E','C3-P','C4')
POSITIONS=('t_U','t_D','t_C','t_A')

def validate_case(case):
    fields={'question_id','question','depth1','depth2','domain','persona','documents'}
    require(isinstance(case,dict) and set(case)==fields,'Expected exactly one case object with the documented fields')
    require(all(isinstance(case[k],str) and case[k].strip() for k in fields-{'documents'}),'Empty or non-string case field')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',case['question_id']),'Invalid question ID')
    d=case['documents'];require(isinstance(d,dict) and set(d)=={'c3_e','c3_p','c4'},'Three reference document variants are required')
    require(all(isinstance(v,str) and v.strip() for v in d.values()),'Empty document')
    return case

def rows_for_case(case,metadata=None):
    validate_case(case);meta=metadata or {}
    variant=meta.get('variant','harmful');require(variant in ('harmful','benign'),'Invalid variant')
    parent=meta.get('parent_question_id',case['question_id'])
    set_id=meta.get('set_id','single');role=meta.get('dev_role','case_study')
    component=meta.get('shared_component_id','single_component')
    out=[]
    for c in CONDITIONS:
        persona=case['persona'] if c in ('C1','C3-E','C3-P','C4') else ''
        key={'C2':'c4','C3-E':'c3_e','C3-P':'c3_p','C4':'c4'}.get(c)
        row=construct_final(dict(question_id=case['question_id'],question=case['question'],condition_id=c,
             persona=persona,document=case['documents'][key] if key else '',repeat_id=0,
             split=role,dev_role=role,parent_question_id=parent,variant=variant,set_id=set_id,
             shared_component_id=component,depth1=case['depth1'],depth2=case['depth2'],domain=case['domain'],
             sample_id=f'{set_id}:{variant}:{case["question_id"]}:{c}'))
        row['historical_C0_label']=meta.get('historical_C0_label')
        row['historical_label_source_sha256']=meta.get('historical_label_source_sha256')
        out.append(row)
    return out

def load_cases(input_path,corpus=False):
    value=read(input_path)
    if not corpus:return rows_for_case(value)
    require(isinstance(value,dict) and set(value)=={'cases'},'Private corpus must contain cases')
    rows=[]
    for record in value['cases']:
        require(set(record)=={'input','metadata'},'Corpus record must have input and metadata')
        required={'variant','parent_question_id','shared_component_id','dev_role','set_id'}
        require(set(record['metadata'])>=required,'Corpus provenance fields missing')
        rows.extend(rows_for_case(record['input'],record['metadata']))
    require(len({r['sample_id'] for r in rows})==len(rows),'Duplicate case/condition within a set')
    by_component={}
    for r in rows:
        k=(r['set_id'],r['shared_component_id']);by_component.setdefault(k,set()).add(r['dev_role'])
    require(all(len(v)==1 for v in by_component.values()),'Shared components cross fit/select/check partitions')
    return rows

def prepare(input_path,run,corpus=False,study='baseline',centers=None,centers_receipt=None,directions=None,directions_receipt=None,cross=None,rubric=None,cohort_mode='fresh_ra'):
    run=Path(run);require(not (run/'manifest.json').exists(),'Run already prepared; use run/status to resume')
    require(not run.exists() or set(p.name for p in run.iterdir())<={'.lock'},'Output directory is not empty')
    rows=load_cases(input_path,corpus)
    if cohort_mode=='historical':
        require(corpus,'Historical cohort needs an explicit private corpus')
        require(all(r.get('historical_C0_label')=='refusal' and r.get('historical_label_source_sha256') for r in rows if r['variant']=='harmful'),'Historical C0 refusal evidence missing')
    from .io import private_artifact
    if study in ('restore','strength','full-vector','all'):require(centers is not None,'Fixed center artifact required for this study')
    if study in ('alignment','all'):require(directions is not None,'Frozen alignment directions required')
    for kind,path,receipt in [('centers',centers,centers_receipt),('directions',directions,directions_receipt)]:
        if path:
            info=private_artifact(path,receipt,kind)
            if kind=='centers':
                from .scoring import load_center_bank
                load_center_bank(path)
            else:
                from .directions import load_directions
                load_directions(path)
            if study!='baseline':require(info.get('scope')=='fit250','Intervention assets must come from frozen fit250')
    if cross:validate_cross(read(cross),rows)
    if rubric:
        from .judging import configure
        configure(rubric)
    run.mkdir(parents=True,exist_ok=True)
    atomic_bytes(run/'input.json',Path(input_path).read_bytes());jsonl(run/'inputs.jsonl',rows)
    for name in ('target','judge','environment'):atomic(run/f'{name}.json',read(ROOT/f'configs/{name}.json'))
    files=['input.json','inputs.jsonl','target.json','judge.json','environment.json']
    assets={}
    from .io import private_artifact
    for kind,p,receipt in [('centers',centers,centers_receipt),('directions',directions,directions_receipt)]:
        if p:
            info=private_artifact(p,receipt,kind)
            if kind=='centers' and study!='baseline':require(info.get('scope')=='fit250','Interventions require the fixed fit250 centers')
            dest=kind+'.npz';atomic_bytes(run/dest,Path(p).read_bytes());atomic(run/(kind+'_receipt.json'),info)
            files.extend([dest,kind+'_receipt.json']);assets[kind]=dest
    if cross:
        validate_cross(read(cross),rows);atomic_bytes(run/'cross.json',Path(cross).read_bytes());files.append('cross.json')
    if rubric:
        atomic_bytes(run/'rubric.json',Path(rubric).read_bytes());files.append('rubric.json')
    if study in ('restore','strength','full-vector','all'):require(centers is not None,'Fixed center artifact required for this study')
    if study in ('alignment','all'):require(directions is not None,'Frozen alignment directions required')
    plan={'study':study,'cohort_mode':cohort_mode,'baseline_tasks':len(rows),'one_case_baseline_tasks':6,'layers':[16,20],
          'intervention_cohort':'fresh_RA_fixed_before_intervention','alphas':[1.0],
          'strength_layer':16,'strength_alphas':[1.5,2.0],'alignment_layer':12,
          'random_seed':20261102,'alignment_random_seed':42,'random_repeats':1,
          'bootstrap_repeats':2000,'bootstrap_seed':20261102,'primary_family':2,
          'main_contrast':['C0','C4'],'gpu_diagnostics_required_each_run':False}
    atomic(run/'protocol.json',plan);files.append('protocol.json')
    m={'version':'1.0.0','created_at':utc(),'code_hash':code_hash(),'files':{n:file_hash(run/n) for n in files},
       'assets':assets,'case_mode':'private_corpus' if corpus else 'single_case','study':study,
       'synthetic_results':False,'historical_results_reproduced':False}
    m['run_identity']=digest(m);atomic(run/'manifest.json',m)
    return {'prepared':True,'rows':len(rows),'study':study,'run_identity':m['run_identity']}

def validate_cross(config,rows):
    require(isinstance(config,dict) and set(config)=={'templates'},'Cross configuration requires templates')
    for row in [r for r in rows if r['condition_id']=='C4']:
        key=f'{row["set_id"]}:{row["variant"]}:{row["question_id"]}'
        t=config['templates'].get(key)
        require(isinstance(t,dict) and set(t)=={'entity_B','purpose_B'},'Supply explicit alternative personas for each cross input')
        require(all(isinstance(v,str) and v.strip() for v in t.values()),'Empty alternative persona')

def cross_rows(rows,config):
    validate_cross(config,rows);out=[]
    index={(r['set_id'],r['variant'],r['question_id'],r['condition_id']):r for r in rows}
    for r in [r for r in rows if r['condition_id']=='C4']:
        key=f'{r["set_id"]}:{r["variant"]}:{r["question_id"]}'
        for axis,mismatch in [('entity','C3-E'),('purpose','C3-P')]:
            for cell in ('AA','AB','BA','BB'):
                original=index[r['set_id'],r['variant'],r['question_id'],'C4' if cell[1]=='A' else mismatch]
                persona=r['persona'] if cell[0]=='A' else config['templates'][key][axis+'_B']
                row={k:v for k,v in original.items() if k not in ('messages','tools','tool_payload','input_hash')}
                row.update(persona=persona,condition_id=axis+'_'+cell,dataset_family='congruence_'+axis,
                           sample_id=key+':'+axis+'_'+cell,cross_axis=axis,cross_cell=cell)
                out.append(construct_final(row))
    return out
