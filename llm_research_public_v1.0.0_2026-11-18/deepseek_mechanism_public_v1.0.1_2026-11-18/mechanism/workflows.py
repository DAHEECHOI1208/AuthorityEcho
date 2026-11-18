"""Analysis recipes over the same saved measurements; no inference or API calls."""
from pathlib import Path
import collections
from .io import require,read_jsonl,read,atomic,jsonl,file_hash


def sensitivity(ledger,out,scope='fit250'):
    from .scoring import fit_centers,score_ledger
    from .readouts import auc_curves
    out=Path(out);require(not out.exists(),'New sensitivity output directory required');out.mkdir(parents=True);report=[]
    for label,conditions in [('four',['C2','C3-E','C3-P','C4']),('five',['C1','C2','C3-E','C3-P','C4']),('six',['C0','C1','C2','C3-E','C3-P','C4'])]:
        folder=out/label;folder.mkdir();center=folder/'centers.npz'
        try:receipt=fit_centers(ledger,center,scope,conditions)
        except ValueError as e:
            report.append({'conditions':conditions,'status':'not_estimable','reason':str(e)});continue
        score_ledger(ledger,center,folder/'scores.jsonl');auc_curves(folder/'scores.jsonl',folder/'auc.csv')
        report.append({'conditions':conditions,'status':'computed','center_sha256':receipt['sha256']})
    atomic(out/'result.json',{'comparisons':report,'scope':scope,'primary_centers_replaced':False,'new_layer_selection':False});return report


def extended_analysis(run,ledger,scores,out):
    """Run data-eligible recipes; record unavailable branches without fabricating values."""
    from .readouts import alert_analysis
    from .regression import regression
    from .representation import validate_f
    from .probes import probe_analysis
    from .alignment_report import alignment_report
    run=Path(run);out=Path(out);require(not out.exists(),'New extended analysis directory required');out.mkdir(parents=True)
    rows=read_jsonl(ledger);score_rows=read_jsonl(scores);report=[]
    for sid in sorted({r['set_id'] for r in rows}):
        folder=out/sid;folder.mkdir();lr=[r for r in rows if r['set_id']==sid];sr=[r for r in score_rows if r['set_id']==sid]
        lp=folder/'ledger.jsonl';sp=folder/'scores.jsonl';jsonl(lp,lr);jsonl(sp,sr)
        frame=Path(str(ledger)+'.frame.json')
        if frame.exists():atomic(str(lp)+'.frame.json',{'rows':[r for r in read(frame)['rows'] if r['set_id']==sid]})
        roles={r['dev_role'] for r in lr};variants={r['variant'] for r in lr}
        sufficient_roles={'calibration_fit','readout_select','readout_check'}<=roles
        if sufficient_roles and variants=={'harmful','benign'}:
            report.append({'set_id':sid,'split_validation':validate_f(lp,folder/'split_validation.json')})
            report.append({'set_id':sid,'full_refit':validate_f(lp,folder/'full_refit.json',positions=('t_U','t_D','t_C','t_A'),scope='full500')})
            report.append({'set_id':sid,'center_sensitivity':sensitivity(lp,folder/'center_sensitivity')})
        else:report.append({'set_id':sid,'center_and_HR_validation':'not_estimable','reason':'Requires separate fit/select/check roles and both harmful and benign groups; one case cannot supply these'})
        if sufficient_roles:
            report.append({'set_id':sid,'probes':probe_analysis(lp,run/'centers.npz',folder/'probes')})
        else:report.append({'set_id':sid,'probes':'not_fitted','reason':'Frozen directions may still be applied; no new independent fitting partitions'})
        if variants=={'harmful','benign'}:
            report.append({'set_id':sid,'adjusted_regression':regression(sp,folder/'regression')})
        else:report.append({'set_id':sid,'adjusted_common_benign_regression':'not_estimable','reason':'No matched benign controls; they are not automatically generated'})
        for mode in ('c4','delta'):
            report.append({'set_id':sid,'alert_mode':mode,**alert_analysis(sp,folder/('alert_'+mode+'.json'),mode=mode)})
    if (run/'directions.npz').exists():report.append({'alignment':alignment_report(run,scores,out/'alignment')})
    atomic(out/'recipe_status.json',{'recipes':report,'all_calculations_use_saved_results':True,'new_target_or_judge_calls':0,
       'missing_support_not_mapped_to_zero':True,'parent_counts_across_sets_not_summed':True})
    return {'recipes':len(report),'status_file':str(out/'recipe_status.json')}
