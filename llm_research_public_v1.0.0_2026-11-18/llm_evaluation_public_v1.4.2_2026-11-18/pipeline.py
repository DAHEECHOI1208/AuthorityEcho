#!/usr/bin/env python3
"""Generate responses for one example, then judge each model and condition."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "target"), str(ROOT / "judges")]
import run_experiment as target
from settings import INPUT_HEADERS
from handoff import CONDITIONS, write_input
from suite import runner
from suite.common import atomic_json, digest, dumps, load_env, read_json, run_lock

VERSION = "1.4.2"
TARGETS = {
    "gpt-4.1-2025-04-14": "config.json",
    "gpt-5.1-2025-11-13": "config.json",
    "gpt-5.4-2026-03-05": "config.json",
    "gpt-5.6-luna": "config.json",
    "claude-opus-4-6": "configs/anthropic.json",
    "gemini-3.1-flash-lite": "configs/gemini.json",
}


def prepare_inputs(path, work):
    raw = path.read_bytes()
    case = json.loads(raw)
    required = {"question_id", "question", "depth1", "depth2", "domain", "persona", "documents"}
    if not isinstance(case, dict) or set(case) != required:
        raise ValueError("Input must be one case object with the documented fields")
    if any(not isinstance(case[k], str) or not case[k].strip() for k in required - {"documents"}):
        raise ValueError("All case metadata, the question and the persona must be nonempty strings")
    docs = case["documents"]
    if (not isinstance(docs, dict) or set(docs) != {"c3_e", "c3_p", "c4"}
            or any(not isinstance(v, str) or not v.strip() for v in docs.values())):
        raise ValueError("documents must contain nonempty c3_e, c3_p and c4 texts")
    qid, prompt = case["question_id"], case["question"]
    mapping = {"schema_version": 2, "questions": {qid: {
        "group_id": "example", "depth1": case["depth1"], "depth2": case["depth2"],
        "domain": case["domain"], "prompt_sha256": digest(prompt)}},
        "groups": {"example": {"persona": case["persona"], "documents": docs}}}
    book = Workbook()
    sheet = book.active
    sheet.title = "input"
    sheet.append(INPUT_HEADERS)
    sheet.append([qid, case["depth1"], case["depth2"], case["domain"], prompt])
    input_file = work / "input.xlsx"
    book.save(input_file)
    book.close()
    return input_file, dumps(mapping).encode("utf-8"), raw


def judge_config(args):
    cfg = read_json(args.judge_config)
    if args.judges:
        ids = args.judges
        if len(set(ids)) != len(ids) or set(ids) - set(cfg["enabled"]):
            raise ValueError("--judges contains a duplicate or unavailable judge")
        cfg["enabled"] = [j for j in cfg["enabled"] if j in ids]
    return runner.validate_config(cfg)


def identity(target_dir):
    parent = read_json(target_dir / "run_manifest.json")
    ids = {c: read_json(target_dir / "conditions" / c / "run_manifest.json")["run_id"]
           for c in parent["condition_order"]}
    return {"model": parent["config"]["model"], "target_run_id": parent["run_id"], "condition_run_ids": ids,
            "input_sha256": parent["input_sha256"], "mapping_sha256": parent["reference_mapping_sha256"]}


def target_configs(args):
    if args.target_config:
        cfg = target.read_config(args.target_config, args.target_model)
        name = cfg["model"]
        if name not in TARGETS:
            raise ValueError("The target configuration must select one of the six commercial target IDs")
        native = read_json(target.ROOT / TARGETS[name])["provider"]
        if cfg["provider"] != native:
            raise ValueError(f"The experiment uses the native {native} provider for {name}")
        configs = [cfg]
    else:
        names = [args.target_model] if args.target_model else TARGETS
        configs = [target.read_config(target.ROOT / TARGETS[name], name) for name in names]
    for cfg in configs:
        cfg["expected_rows"] = 1
    return configs


def start(args, out):
    judge = judge_config(args)
    configs = target_configs(args)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        temp = Path(tmp)
        data, mapping, case_bytes = prepare_inputs(args.input, temp)
        publish = temp / "new_run"
        identities = []
        for cfg in configs:
            target_dir = publish / "targets" / cfg["model"]
            target.prepare_all(target_dir, data, cfg, mapping)
            identities.append(identity(target_dir))
        manifest = {"version": VERSION, "targets": identities, "judge_config": judge,
                    "input_case_sha256": digest(case_bytes)}
        (publish / "input.json").write_bytes(case_bytes)
        marker = publish / "fresh_run.json"
        atomic_json(marker, manifest)
        (publish / "fresh_run.sha256").write_text(digest(marker.read_bytes()) + "\n", encoding="ascii")
        publish.rename(out)


def run_target(out, record, args):
    folder = out / "targets" / record["model"]
    with target.e.run_lock(folder):
        if identity(folder) != record:
            raise ValueError("Target journals do not belong to this fresh run")
        cfg = target.validate_all(folder)
        retry_errors = getattr(args, "retry_target_errors", False)
        retry_unknown = getattr(args, "retry_target_unknown", False)
        if (retry_errors or retry_unknown) and (out / "judge").exists():
            raise ValueError("Judge input already frozen; start a new run for new target answers")
        counts = target.write_summary(folder)["conditions"]
        pending = sum(c["unattempted"] + c["status_counts"].get("RUNNING", 0) for c in counts.values())
        # A fatal provider error can occur on the very last row. Respect it on resume as well.
        fatal = False
        if not pending and not (out / "judge").exists():
            for cond in counts:
                db = target.e.connect(folder / "conditions" / cond)
                try:
                    fatal = fatal or bool(target.latest_fatal(db))
                finally:
                    db.close()
        if pending or fatal or retry_errors or retry_unknown:
            key = os.environ.get(cfg["api_key_env"], "").strip()
            if cfg["api_key_env"] and not key:
                raise ValueError(f"Set {cfg['api_key_env']} for the target, then resume this run")
            code = target.execute_all(folder, cfg, key, retry_errors, retry_unknown)
            if code:
                raise RuntimeError(f"{cfg['model']} stopped. See {folder}/execution_events.jsonl")


def judge_fresh(out, manifest):
    judge = out / "judge"
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        staging = Path(tmp)
        inputs = {"conditions": list(CONDITIONS.values()), "target_metadata": True, "files": []}
        source_hashes = []
        for record in manifest["targets"]:
            name = record["model"]
            folder = out / "targets" / name
            with target.e.run_lock(folder):
                if identity(folder) != record:
                    raise ValueError(f"{name}: target run changed before judge handoff")
                source, data = write_input(folder, staging, name + ".jsonl")
            if (source["run_id"], source["target_model"]) != (record["target_run_id"], name):
                raise ValueError(f"{name}: source identity mismatch")
            inputs["files"].append({"path": data.name, "target_model": name, "run_id": source["run_id"]})
            source_hashes.append(digest(data.read_bytes()))
        if judge.exists():
            frozen = runner.verify_run(judge)
            if (frozen["inputs"]["manifest"] != inputs or frozen["config"] != manifest["judge_config"]
                    or [s["sha256"] for s in frozen["inputs"]["source_files"]] != source_hashes):
                raise ValueError("Saved judge snapshot differs from this run")
        else:
            atomic_json(staging / "inputs.json", inputs)
            atomic_json(staging / "judges.json", manifest["judge_config"])
            runner.prepare(staging / "inputs.json", staging / "judges.json", judge)
    previous_export = (judge / "export_state.json").stat().st_mtime_ns
    outcome = runner.execute(judge)
    state = read_json(judge / "export_state.json")
    print(dumps(outcome))
    if (previous_export == (judge / "export_state.json").stat().st_mtime_ns or outcome["stopped"]
            or state["pending_results"] or not state["full_analysis_current"]):
        raise RuntimeError("Judge run or export incomplete. See judge/logs/events.jsonl")
    print(judge / "results.xlsx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("run", help="Generate all selected target responses anew, then batch judge")
    new.add_argument("--input", default=ROOT / "input.json", type=Path, help="JSON file containing exactly one case")
    new.add_argument("--target-config", type=Path, help="Override native settings for one commercial target")
    new.add_argument("--target-model", choices=tuple(TARGETS), help="Only this target; default is all six in the configured order")
    new.add_argument("--judge-config", default=ROOT / "judges/configs/judges.json", type=Path)
    new.add_argument("--judges", nargs="+", help="Judge IDs; defaults to all four")
    resume = commands.add_parser("resume", help="Finish a previously started v1.4.2 run")
    resume.add_argument("--retry-target-errors", action="store_true")
    resume.add_argument("--retry-target-unknown", action="store_true", help="Potential duplicate provider delivery")
    for cmd in (new, resume):
        cmd.add_argument("--out", required=True, type=Path)
        cmd.add_argument("--env", default=".env")
    args = parser.parse_args()
    out = args.out.resolve()
    load_env(args.env)
    if args.command == "run":
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise ValueError("Run directory already exists. Choose a new --out for fresh target calls.")
        start(args, out)
    if not (out / "fresh_run.json").is_file():
        raise ValueError("No fresh_run.json from this package in --out")
    with run_lock(out):
        marker = out / "fresh_run.json"
        if digest(marker.read_bytes()) != (out / "fresh_run.sha256").read_text(encoding="ascii").strip():
            raise ValueError("Fresh run manifest changed")
        manifest = read_json(marker)
        if (manifest.get("version") != VERSION or manifest.get("input_case_sha256") != digest((out / "input.json").read_bytes())):
            raise ValueError("This is not a compatible fresh run")
        for record in manifest["targets"]:
            print(f"Target {record['model']} / {record['target_run_id']}", flush=True)
            run_target(out, record, args)
        judge_fresh(out, manifest)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
