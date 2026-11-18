#!/usr/bin/env python3
"""C0 -> C1 -> C2 -> C3-E -> C3-P -> C4. prepare/export/status are offline."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import uuid
import engine as e
import providers
from settings import CONDITIONS, validate_config

ROOT=Path(__file__).resolve().parent

def condition_folder(folder,condition): return Path(folder)/'conditions'/condition

def read_config(path,model=None):
    cfg=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if model is not None: cfg['model']=model
    return validate_config(cfg)

def prepare_all(folder,source,cfg,mapping_bytes):
    folder=Path(folder);source=Path(source)
    folder.mkdir(parents=True,exist_ok=True)
    if any(p.name!='.run.lock' for p in folder.iterdir()):
        raise ValueError('Use a new run directory. Resume an existing run with its execution command.')
    # Validate ALL mappings and payload shapes before preparing any condition.
    mapping=json.loads(mapping_bytes)
    for condition in CONDITIONS:
        cc={**cfg,'condition':condition}
        _,rows=e.read_input(source,cc,mapping)
        for row in rows: e.payload_for(cc,row)
    manifest={'schema_version':3,'run_id':str(uuid.uuid4()),'ready':False,
              'created_at_utc':e.utc_now(),'condition_order':list(CONDITIONS),
              'config':cfg,'provider_settings':providers.describe_settings(cfg),'input_sha256':e.sha(source.read_bytes()),
              'reference_mapping_sha256':e.sha(mapping_bytes),
              'runtime':e.runtime_info(),'questions_per_condition':cfg['expected_rows'],
              'planned_results':cfg['expected_rows']*cfg['repetitions']*len(CONDITIONS)}
    e.atomic_json(folder/'run_manifest.json',manifest)
    e.atomic_json(folder/'config_frozen.json',cfg)
    e.atomic_bytes(folder/'input_original.xlsx',source.read_bytes())
    e.atomic_bytes(folder/'reference_mapping_original.json',mapping_bytes)
    for condition in CONDITIONS:
        e.prepare(condition_folder(folder,condition),source,{**cfg,'condition':condition},mapping_bytes)
    manifest['ready']=True
    e.atomic_json(folder/'run_manifest.json',manifest)
    write_summary(folder)
    return manifest

def validate_all(folder,cfg=None,source=None):
    folder=Path(folder)
    manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('ready') or manifest.get('condition_order')!=list(CONDITIONS):
        raise ValueError('Preparation is incomplete or the execution order changed. Use a new run directory.')
    frozen=manifest['config'];validate_config(frozen)
    if cfg is not None and cfg!=frozen:
        raise ValueError('Model or settings changed. Start in a new run directory.')
    if json.loads((folder/'config_frozen.json').read_text(encoding='utf-8'))!=frozen:
        raise ValueError('The frozen configuration file changed.')
    for filename,hashkey in [('input_original.xlsx','input_sha256'),('reference_mapping_original.json','reference_mapping_sha256')]:
        if e.sha((folder/filename).read_bytes())!=manifest[hashkey]:
            raise ValueError(filename+' hash mismatch; stopping execution.')
    if source is not None and e.sha(Path(source).read_bytes())!=manifest['input_sha256']:
        raise ValueError('The input file differs from the existing run.')
    if e.runtime_info()['runner_sha256']!=manifest['runtime']['runner_sha256']:
        raise ValueError('Execution code changed after preparation. Use a new run directory.')
    for condition in CONDITIONS:
        cf=condition_folder(folder,condition)
        if not (cf/'journal.sqlite3').is_file(): raise ValueError(condition+' journal is missing.')
        db=e.connect(cf)
        try: e.validate_run(db,cf,{**frozen,'condition':condition})
        finally: db.close()
    return frozen

def root_event(folder,kind,**fields):
    with open(Path(folder)/'execution_events.jsonl','ab') as f:
        f.write((e.dumps({'timestamp_utc':e.utc_now(),'kind':kind,**fields})+'\n').encode('utf-8'))
        f.flush();os.fsync(f.fileno())

def write_summary(folder):
    summary={'updated_at_utc':e.utc_now(),'condition_order':list(CONDITIONS),'conditions':{}}
    for condition in CONDITIONS:
        cf=condition_folder(folder,condition)
        db=e.connect(cf)
        try:
            counts={r[0]:r[1] for r in db.execute('SELECT status,COUNT(*) FROM repetitions GROUP BY status')}
            summary['conditions'][condition]={'status_counts':counts,
                'attempts':db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],
                'unattempted':counts.get('PENDING',0),
                'result_file':e.result_filename(e.get_meta(db,'config'))}
        finally: db.close()
    e.atomic_json(Path(folder)/'summary.json',summary)
    return summary

def latest_fatal(db):
    rows=db.execute('SELECT a.result_json FROM repetitions r JOIN attempts a ON a.attempt_id=r.latest_attempt_id')
    return [json.loads(row[0]) for row in rows if row[0] and json.loads(row[0]).get('stop_run')]

def execute_all(folder,cfg,key,retry_errors=False,retry_unknown=False,transport=None,wait=e.responsive_wait):
    """All rows attempted/recorded before advancing; successes are never resent."""
    # Recover every durable receipt first, including a crash immediately after a fatal response.
    for condition in CONDITIONS:
        cf=condition_folder(folder,condition);db=e.connect(cf)
        try:
            e.recover(db,cf)
            fatal=latest_fatal(db)
            if fatal and not retry_errors:
                raise ValueError(f'{condition}: A provider-level error stopped this run. Resolve it and explicitly retry target errors.')
        finally: db.close()
    root_event(folder,'run_started',provider=cfg['provider'],model=cfg['model'],retry_errors=retry_errors,retry_unknown=retry_unknown)
    outcomes={};code=0
    try:
        for condition in CONDITIONS:
            cf=condition_folder(folder,condition);db=e.connect(cf)
            try:
                cc={**cfg,'condition':condition}
                print(f'\n[{condition}] {cfg["provider"]} / {cfg["model"]}',flush=True)
                root_event(folder,'condition_started',condition=condition)
                outcome=e.run_pending(db,cf,cc,key,retry_errors=retry_errors,retry_unknown=retry_unknown,
                                      transport=transport or providers.transport_for(cc),wait=wait)
                outcomes[condition]=outcome
                root_event(folder,'condition_finished',condition=condition,**outcome)
                if outcome['interrupted']: code=130;break
                if outcome['stopped_for_error']: code=2;break
                if db.execute("SELECT COUNT(*) FROM repetitions WHERE status IN ('PENDING','RUNNING')").fetchone()[0]:
                    raise RuntimeError('The preceding condition has pending items. Complete it before continuing.')
            finally: db.close()
            # Also respect the interval across condition boundaries.
            if condition!=CONDITIONS[-1] and outcome['processed']: wait(cfg['min_interval_seconds'])
    except KeyboardInterrupt:
        code=130
        root_event(folder,'user_interrupted')
    except Exception as exc:
        code=1
        root_event(folder,'local_execution_error',error_type=type(exc).__name__,error_message=e.redact(str(exc),key))
        raise
    finally:
        write_summary(folder)
        root_event(folder,'run_finished',exit_code=code,outcomes=outcomes)
    return code

def export_all_conditions(folder):
    for condition in CONDITIONS:
        cf=condition_folder(folder,condition);db=e.connect(cf)
        try:
            e.recover(db,cf)
            e.export_all(db,cf,tolerate_excel_lock=True)
        finally: db.close()
    return write_summary(folder)

