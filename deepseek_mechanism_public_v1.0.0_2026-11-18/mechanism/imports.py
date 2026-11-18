"""Read-only import of explicit artifacts and the original identity-indexed cache."""
from pathlib import Path
import numpy as np
from .io import require,read,read_jsonl,file_hash,atomic,npz_save,jsonl
from .scoring import load_center_bank,POSITIONS


def register_artifact(path,out,kind,scope,expected_sha256,evidence):
    path=Path(path);require(file_hash(path)==expected_sha256,'Artifact differs from the supplied frozen hash')
    source=read(evidence)
    require(source.get('scope')==scope and source.get('kind')==kind,'Evidence must explicitly identify artifact kind and fitting scope')
    require(source.get('sha256')==expected_sha256,'Evidence hash does not bind this artifact')
    if kind=='centers':
        bank=load_center_bank(path);require(np.isfinite(bank[:,3]).all(),'t_A center vectors missing')
    else:
        from .directions import load_directions
        load_directions(path)
    receipt={**source,'evidence_sha256':file_hash(evidence),'source_artifact_sha256':file_hash(path),'private_path_not_copied_into_public_package':True}
    atomic(out,receipt);return {'verified':True,'sha256':expected_sha256,'scope':scope}


def import_cache(cache,identity,metadata,out):
    cache=Path(cache);ident=read(identity);out=Path(out);require(not out.exists(),'Import output must be new')
    require(ident.get('cache_sha256')==file_hash(cache),'Original cache hash mismatch')
    spec=ident['identity'];x=np.load(cache,mmap_mode='r',allow_pickle=False)
    require(list(x.shape)==spec['shape'] and x.dtype==np.float32 and x.shape[1:]==(27,4,2048),'Original cache axes/dtype differ')
    ix=spec['index'];require(len(ix)==len(x) and len({r['sample_id'] for r in ix})==len(ix),'Duplicate or incomplete cache index')
    meta=read_jsonl(metadata);by={r['sample_id']:r for r in meta};require(len(by)==len(meta),'Duplicate metadata sample')
    required={'sample_id','question_id','parent_question_id','set_id','variant','condition_id','dev_role','shared_component_id','group','npz_sha256'}
    require(all(set(r)>=required for r in meta),'Explicit semantic metadata fields missing; do not infer labels from row order')
    out.mkdir(parents=True);rows=[]
    for i,item in enumerate(ix):
        sid=item['sample_id'];require(sid in by,'Cache sample missing from semantic metadata: '+sid);r=by[sid]
        require(item['npz_sha256']==r['npz_sha256'],'Original NPZ provenance differs')
        a=np.array(x[i],copy=True);require(not np.isinf(a).any(),'Infinite cached activation')
        tid='import_'+str(i);dest=out/'activations'/f'{tid}.npz';npz_save(dest,activations=a,layers=np.arange(27),positions=np.asarray(POSITIONS))
        rows.append({**r,'task_id':tid,'arm':'baseline','layer':None,'alpha':None,'activation_path':str(dest.resolve()),'activation_sha256':file_hash(dest),
          'capture_mode':'original_cache_identity_verified','source_cache_row':i,'source_identity_sha256':file_hash(identity),'strict_valid':r['group'] in ('HR','HA','BR','BA')})
    jsonl(out/'ledger.jsonl',rows);atomic(out/'import_receipt.json',{'cache_sha256':file_hash(cache),'identity_sha256':file_hash(identity),'metadata_sha256':file_hash(metadata),'rows':len(rows),'source_modified':False})
    return {'rows':len(rows),'ledger':str(out/'ledger.jsonl'),'no_generation':True}


def import_reference(source_dir,out):
    """Import the known private reference assets without changing their contents."""
    from .io import ROOT,atomic_bytes
    from .judging import export_rubric
    source_dir=Path(source_dir);out=Path(out);require(not out.exists(),'Reference output directory already exists')
    contract=read(ROOT/'configs/reference_identity.json');center=source_dir/'reference/centers_fit250_all27.npz'
    require(file_hash(center)==contract['fit250_centers_sha256'],'Original fit250 center hash differs')
    require(read(source_dir/'configs/target.json')==read(ROOT/'configs/target.json'),'Original target configuration differs')
    require(read(source_dir/'configs/judge.json')==read(ROOT/'configs/judge.json'),'Original Judge configuration differs')
    assignments=source_dir/'reference/split_assignments500.jsonl'
    original=read(source_dir/'reference/asset_contract.json')
    require(original['centers_sha256']==file_hash(center) and original['assignments_sha256']==file_hash(assignments),'Original reference asset contract failed')
    out.mkdir(parents=True);atomic_bytes(out/'centers.npz',center.read_bytes())
    metadata=read_jsonl(assignments);fit=[r for r in metadata if r.get('dev_role')=='calibration_fit']
    atomic(out/'centers_receipt.json',{'kind':'centers','scope':'fit250','sha256':file_hash(center),
      'source_asset_contract_sha256':file_hash(source_dir/'reference/asset_contract.json'),
      'fit_component_ids':sorted({r['shared_component_id'] for r in fit}),
      'fit_parent_ids':sorted({r.get('parent_question_id',r.get('question_id')) for r in fit}),
      'center_recomputed':False})
    export_rubric(source_dir/'vendor/r12/P10_judge.py',out/'rubric.json')
    return {'centers':str(out/'centers.npz'),'receipt':str(out/'centers_receipt.json'),'rubric':str(out/'rubric.json'),'private_assets_do_not_publish':True}


def import_alignment_reference(source_dir,out):
    from .io import ROOT,atomic_bytes
    from .directions import load_directions
    source_dir=Path(source_dir);out=Path(out);require(not out.exists(),'Reference output already exists')
    path=source_dir/'data/readout.npz';expected=read(ROOT/'configs/reference_identity.json')['shared_readout_sha256']
    require(file_hash(path)==expected,'Original purpose shared readout hash differs');load_directions(path)
    out.mkdir(parents=True);atomic_bytes(out/'directions.npz',path.read_bytes())
    atomic(out/'directions_receipt.json',{'kind':'directions','scope':'fit250','sha256':expected,'direction_definition':'original shared congruence directions at layers 12 to 16; no entity-specific refit','center_recomputed':False})
    return {'directions':str(out/'directions.npz'),'receipt':str(out/'directions_receipt.json'),'source_modified':False}
