"""One entry point for generation, measurement, interventions and saved-result analysis."""
import argparse
import json
import os
from pathlib import Path
from .io import ROOT,read,require,load_run,lock,atomic,file_hash,utc


def analysis_bindings(run):
    run=Path(run)
    files=[p for folder in ('outputs','judges','extracted') for p in (run/folder).rglob('*.json')]
    return {str(p.relative_to(run)):file_hash(p) for p in sorted(files)}


def pipeline(run,until='complete'):
    from . import runtime,judging,interventions,gates,operations,scoring,analysis
    run=Path(run);m,rows=load_run(run);study=m['study']
    if until not in ('smoke','baseline','extract'):
        judging.configure(run/'rubric.json')
        require(bool(os.environ.get(read(run/'judge.json')['api_key_env'])),'Missing configured Judge API key environment variable')
    def stage(name,fn):
        print('START',name,flush=True);result=operations.run_stage(run,name,fn);print(json.dumps(result,ensure_ascii=True,default=str),flush=True);print('COMPLETE',name,flush=True)
    if not (run/'smoke.json').exists():stage('smoke',lambda:runtime.smoke(run))
    else:require(read(run/'smoke.json')['passed'],'Prior baseline smoke did not pass')
    if until=='smoke':return {'through':until}
    stage('baseline',lambda:runtime.baseline(run))
    if until=='baseline':return {'through':until}
    stage('extract',lambda:runtime.extract(run))
    if until=='extract':return {'through':until}
    if (run/'cross.json').exists():stage('cross-baseline',lambda:runtime.baseline(run,cross=True))
    stage('judge-baseline',lambda:judging.evaluate(run));stage('baseline-gate',lambda:gates.baseline_gate(run))
    stage('freeze',lambda:interventions.freeze(run))
    if until=='freeze':return {'through':until}
    if study in ('restore','strength','full-vector','all'):
        cohort=interventions.load_cohort(run);plan=interventions.plans(rows,cohort,study)
        smoke_n=len(plan)//max(1,len(cohort.get('generation_members',cohort['members'])))
        if not (run/'intervention_smoke.json').exists():
            stage('intervention-smoke',lambda:interventions.run_interventions(run,limit=smoke_n))
            stage('judge-intervention-smoke',lambda:judging.evaluate(run))
            stage('intervention-smoke-gate',lambda:gates.intervention_gate(run,smoke=True))
        else:require(read(run/'intervention_smoke.json')['passed'],'Prior intervention smoke did not pass')
        if until=='intervention-smoke':return {'through':until}
        stage('intervention',lambda:interventions.run_interventions(run));stage('judge-intervention',lambda:judging.evaluate(run))
        stage('intervention-gate',lambda:gates.intervention_gate(run,smoke=False))
    if study in ('alignment','all'):
        stage('alignment-restore',lambda:interventions.alignment_restore(run));stage('judge-alignment',lambda:judging.evaluate(run))
    batch='main'
    if study in ('strength','all'):
        batch='strength_contemporary';stage('judge-contemporary-comparators',lambda:judging.evaluate(run,batch=batch))
    if until=='judge':return {'through':until}
    out=run/'analysis'
    # Only a verified complete analysis is reused. A crashed partial output is retained.
    if (out/'complete.json').exists():
        receipt=read(out/'complete.json');require(all(file_hash(out/r)==h for r,h in receipt['files'].items()),'Analysis receipt changed')
        require(receipt['source_bindings']==analysis_bindings(run),'Analysis inputs changed; preserve prior results and use a new analysis path')
    else:
        require(not out.exists(),'Partial analysis directory exists; preserve it and use analyze --out with a new path')
        stage('analyze',lambda:analysis.analyze(run,out,batch))
        stage('prefix-audit',lambda:__import__('mechanism.diagnostics',fromlist=['prefix_audit']).prefix_audit(run,out/'prefix_diagnostic.json'))
        stage('export-ledger',lambda:{'rows':len(scoring.export_ledger(run,out/'activation_ledger.jsonl',batch))})
        if (run/'centers.npz').exists():
            stage('score',lambda:scoring.score_ledger(out/'activation_ledger.jsonl',run/'centers.npz',out/'scores.jsonl',run/'directions.npz' if (run/'directions.npz').exists() else None))
            from .representation import linkage
            from .readouts import common_contrasts,cross_report,auc_curves
            stage('linkage',lambda:linkage(out/'scores.jsonl',out/'linkage'))
            stage('common-cohort',lambda:common_contrasts(out/'scores.jsonl',out/'common_cohort'))
            stage('readouts',lambda:auc_curves(out/'scores.jsonl',out/'readout_auc.csv'))
            if (run/'cross.json').exists():stage('cross-analysis',lambda:cross_report(out/'scores.jsonl',out/'cross'))
            if study=='all':
                from .workflows import extended_analysis
                stage('extended-analysis',lambda:extended_analysis(run,out/'activation_ledger.jsonl',out/'scores.jsonl',out/'extended'))
        atomic(out/'complete.json',{'run_identity':m['run_identity'],'created_at':utc(),'source_bindings':analysis_bindings(run),'files':{str(p.relative_to(out)):file_hash(p) for p in sorted(out.rglob('*')) if p.is_file()},'gpu_efficacy_claim':False})
    return {'completed':True,'analysis':str(out),'study':study,'no_automatic_scientific_validation':True}


def parser():
    p=argparse.ArgumentParser(description='Pinned DeepSeek mechanism study, with private external data and English public source.')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--input',required=True);q.add_argument('--run',required=True);q.add_argument('--corpus',action='store_true');q.add_argument('--cohort-mode',choices=['fresh_ra','historical'],default='fresh_ra');q.add_argument('--study',choices=['baseline','restore','strength','full-vector','alignment','all'],default='baseline')
    for name in ('centers','centers-receipt','directions','directions-receipt','cross','rubric'):q.add_argument('--'+name)
    q=sub.add_parser('export-rubric');q.add_argument('--source',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('import-reference');q.add_argument('--source-dir',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('import-alignment-reference');q.add_argument('--source-dir',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('preflight');q.add_argument('--gpu',action='store_true')
    for cmd in ('run','status','smoke','baseline','extract','judge','freeze','intervene','alignment','baseline-gate','intervention-gate'):
        q=sub.add_parser(cmd);q.add_argument('--run',required=True)
        if cmd=='run':q.add_argument('--until',choices=['smoke','baseline','extract','freeze','intervention-smoke','judge','complete'],default='complete')
        if cmd in ('baseline','intervene','judge'):q.add_argument('--limit',type=int)
        if cmd=='baseline':q.add_argument('--cross',action='store_true')
        if cmd=='judge':q.add_argument('--batch',default='main');q.add_argument('--resume-uncertain',action='store_true')
        if cmd=='alignment':q.add_argument('--axis',choices=['entity','purpose','both'],default='both')
    for cmd in ('analyze','export-ledger','prefix-audit','backup','human-review','temporal-report','diagnose'):
        q=sub.add_parser(cmd);q.add_argument('--run',required=True);q.add_argument('--out',required=True)
        if cmd in ('analyze','export-ledger','human-review'):q.add_argument('--batch',default='main')
        if cmd=='human-review':q.add_argument('--review-file')
        if cmd=='temporal-report':q.add_argument('--first',required=True);q.add_argument('--second',required=True)
        if cmd=='diagnose':q.add_argument('--mode',choices=['repeat','layer0','length','o-proj'],default='repeat');q.add_argument('--sample-id');q.add_argument('--repeats',type=int,default=3)
    for cmd in ('retry-target','retry-judge'):
        q=sub.add_parser(cmd);q.add_argument('--run',required=True);q.add_argument('--task-id',required=True)
        if cmd=='retry-judge':q.add_argument('--batch',default='main')
    q=sub.add_parser('audit-npz');q.add_argument('--path',required=True);q.add_argument('--out',required=True)
    for cmd in ('fit-centers','fit-probes','validate-f','probe-analysis','score','center-sensitivity'):
        q=sub.add_parser(cmd);q.add_argument('--ledger',required=True);q.add_argument('--out',required=True)
        if cmd=='center-sensitivity':q.add_argument('--scope',choices=['fit250','full500'],default='fit250')
        if cmd=='fit-centers':q.add_argument('--scope',choices=['fit250','full500'],default='fit250');q.add_argument('--conditions',nargs='+',default=['C2','C3-E','C3-P','C4'])
        if cmd=='validate-f':q.add_argument('--scope',choices=['split','full500'],default='split');q.add_argument('--positions',nargs='+',choices=['t_U','t_D','t_C','t_A'],default=['t_C','t_A'])
        if cmd in ('score','probe-analysis'):q.add_argument('--centers',required=True)
        if cmd=='score':q.add_argument('--directions')
    for cmd in ('linkage','common-cohort','readouts','cross-report','regression','alerts'):
        q=sub.add_parser(cmd);q.add_argument('--scores',required=True);q.add_argument('--out',required=True)
        if cmd=='alerts':
            q.add_argument('--layer',type=int,default=16);q.add_argument('--position',default='t_A');q.add_argument('--metric',choices=['f','g','a'],default='g');q.add_argument('--mode',choices=['c4','delta'],default='c4');q.add_argument('--budget',type=float,default=.05)
    q=sub.add_parser('register-artifact');q.add_argument('--path',required=True);q.add_argument('--out',required=True);q.add_argument('--kind',choices=['centers','directions'],required=True);q.add_argument('--scope',choices=['fit250','full500'],required=True);q.add_argument('--expected-sha256',required=True);q.add_argument('--evidence',required=True)
    q=sub.add_parser('import-cache');q.add_argument('--cache',required=True);q.add_argument('--identity',required=True);q.add_argument('--metadata',required=True);q.add_argument('--out',required=True)
    for cmd in ('alignment-report','extended-analysis'):
        q=sub.add_parser(cmd);q.add_argument('--run',required=True);q.add_argument('--scores',required=True);q.add_argument('--out',required=True)
        if cmd=='extended-analysis':q.add_argument('--ledger',required=True)
    return p


def dispatch(a):
    c=a.command
    if c=='prepare':
        from .inputs import prepare
        return prepare(a.input,a.run,a.corpus,a.study,a.centers,a.centers_receipt,a.directions,a.directions_receipt,a.cross,a.rubric,a.cohort_mode)
    if c=='export-rubric':
        from .judging import export_rubric
        return export_rubric(a.source,a.out)
    if c=='import-alignment-reference':
        from .imports import import_alignment_reference
        return import_alignment_reference(a.source_dir,a.out)
    if c=='import-reference':
        from .imports import import_reference
        return import_reference(a.source_dir,a.out)
    if c=='preflight':
        from .runtime import environment_check,verify_snapshot
        if a.gpu:return {'environment':environment_check(),'weight_hashes':verify_snapshot()}
        import importlib.metadata
        return {'mode':'CPU_only','numpy':importlib.metadata.version('numpy'),'target':read(ROOT/'configs/target.json'),'gpu_runtime_validated':False,'api_called':False}
    if c=='alignment-report':
        from .alignment_report import alignment_report
        return alignment_report(a.run,a.scores,a.out)
    if c=='extended-analysis':
        from .workflows import extended_analysis
        return extended_analysis(a.run,a.ledger,a.scores,a.out)
    if c=='run':return pipeline(a.run,a.until)
    if c in ('status','retry-target','retry-judge','backup','human-review','temporal-report','audit-npz'):
        from . import operations as op
        if c=='status':return op.status(a.run)
        if c=='retry-target':return op.retry_target(a.run,a.task_id)
        if c=='retry-judge':return op.retry_judge(a.run,a.batch,a.task_id)
        if c=='backup':return op.backup(a.run,a.out)
        if c=='human-review':return op.human_review(a.run,a.out,a.review_file,a.batch)
        if c=='temporal-report':return op.temporal_report(a.run,a.first,a.second,a.out)
        return op.audit_npz(a.path,a.out)
    if c in ('smoke','baseline','extract'):
        from . import runtime
        return runtime.baseline(a.run,a.limit,a.cross) if c=='baseline' else getattr(runtime,c)(a.run)
    if c=='judge':
        from .judging import evaluate
        return evaluate(a.run,a.batch,a.resume_uncertain,a.limit)
    if c in ('freeze','intervene','alignment'):
        from . import interventions as i
        return i.freeze(a.run) if c=='freeze' else i.run_interventions(a.run,limit=a.limit) if c=='intervene' else i.alignment_restore(a.run,a.axis)
    if c in ('baseline-gate','intervention-gate'):
        from . import gates
        return gates.baseline_gate(a.run) if c=='baseline-gate' else gates.intervention_gate(a.run,smoke=False)
    if c in ('analyze','export-ledger'):
        if c=='analyze':
            from .analysis import analyze
            return analyze(a.run,a.out,a.batch)
        from .scoring import export_ledger
        return {'rows':len(export_ledger(a.run,a.out,a.batch))}
    if c in ('prefix-audit','diagnose'):
        from .diagnostics import prefix_audit,gpu_diagnostics
        return prefix_audit(a.run,a.out) if c=='prefix-audit' else gpu_diagnostics(a.run,a.out,a.mode,a.sample_id,a.repeats)
    if c in ('fit-centers','fit-probes','score','validate-f','probe-analysis','center-sensitivity'):
        if c=='center-sensitivity':
            from .workflows import sensitivity
            return sensitivity(a.ledger,a.out,a.scope)
        from . import scoring
        if c=='fit-centers':return scoring.fit_centers(a.ledger,a.out,a.scope,a.conditions)
        if c=='fit-probes':return scoring.fit_probes(a.ledger,a.out)
        if c=='score':return scoring.score_ledger(a.ledger,a.centers,a.out,a.directions)
        if c=='validate-f':
            from .representation import validate_f
            return validate_f(a.ledger,a.out,a.positions,a.scope)
        from .probes import probe_analysis
        return probe_analysis(a.ledger,a.centers,a.out)
    if c in ('register-artifact','import-cache'):
        from .imports import register_artifact,import_cache
        return register_artifact(a.path,a.out,a.kind,a.scope,a.expected_sha256,a.evidence) if c=='register-artifact' else import_cache(a.cache,a.identity,a.metadata,a.out)
    if c=='linkage':
        from .representation import linkage
        return linkage(a.scores,a.out)
    if c=='regression':
        from .regression import regression
        return regression(a.scores,a.out)
    from . import readouts
    if c=='common-cohort':return readouts.common_contrasts(a.scores,a.out)
    if c=='readouts':return readouts.auc_curves(a.scores,a.out)
    if c=='cross-report':return readouts.cross_report(a.scores,a.out)
    if c=='alerts':return readouts.alert_analysis(a.scores,a.out,a.layer,a.position,a.metric,a.mode,a.budget)
    raise ValueError('Unknown command')


def main():
    a=parser().parse_args()
    if hasattr(a,'out'):require(not Path(a.out).exists(),'Output already exists; preserve it and choose a new path')
    if hasattr(a,'run') and a.command!='status':
        with lock(a.run):r=dispatch(a)
    else:r=dispatch(a)
    print(json.dumps(r,ensure_ascii=True,indent=2,default=str))
