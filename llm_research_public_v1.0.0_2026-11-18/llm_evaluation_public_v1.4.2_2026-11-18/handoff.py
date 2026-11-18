"""Load the latest attempt for each question generated in this run."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

CONDITIONS = {"c0": "C0", "c1": "C1", "c2": "C2", "c3_e": "C3-E", "c3_p": "C3-P", "c4": "C4"}
STRATA = ["depth1", "depth2", "domain", "context_group"]
QUERY = """
    SELECT q.question_id, q.prompt, q.prompt_sha256, q.values_json, r.repeat_index,
           r.status, r.latest_attempt_id, a.question_id AS attempt_question,
           a.repeat_index AS attempt_repeat, a.result_json
    FROM questions q JOIN repetitions r USING (question_id)
    LEFT JOIN attempts a ON a.attempt_id = r.latest_attempt_id
    ORDER BY q.ordinal, r.repeat_index
"""


def read_condition(root, parent, name):
    folder = root / "conditions" / name
    journal = folder / "journal.sqlite3"
    child = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    with closing(sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        meta = {row[0]: json.loads(row[1]) for row in db.execute("SELECT key, value FROM meta")}
        cfg = dict(parent["config"], condition=name)
        if (not meta.get("ready") or meta["config"] != cfg or child["run_id"] != meta["run_id"]
                or child["config"] != cfg or meta["schema_version"] != 3):
            raise ValueError(f"{name}: inconsistent condition journal/manifest")
        for key in ("input_sha256", "reference_mapping_sha256"):
            if child[key] != meta[key] or parent[key] != meta[key]:
                raise ValueError(f"{name}: different target inputs ({key})")
        count = 0
        for row in db.execute(QUERY):
            qid, rep, prompt = row["question_id"], row["repeat_index"], row["prompt"]
            if row["status"] in {"PENDING", "RUNNING"}:
                raise ValueError(f"{name}/{qid}/r{rep}: target is {row['status']}; judge snapshot requires a finished run")
            if row["attempt_question"] != qid or row["attempt_repeat"] != rep or not row["result_json"]:
                raise ValueError(f"{name}/{qid}/r{rep}: missing or mismatched latest attempt")
            result = json.loads(row["result_json"])
            if (result["question_id"], result["repeat_index"], result["attempt_id"], result["status"]) != (
                qid, rep, row["latest_attempt_id"], row["status"]
            ):
                raise ValueError(f"{name}/{qid}/r{rep}: latest result identity mismatch")
            values = json.loads(row["values_json"])
            if len(values) != 8 or values[0] != qid or values[4] != prompt:
                raise ValueError(f"{name}/{qid}: question metadata mismatch")
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != row["prompt_sha256"]:
                raise ValueError(f"{name}/{qid}: question hash mismatch")
            if not isinstance(result["answer"], str):
                raise ValueError(f"{name}/{qid}: response must be text")
            count += 1
            yield {
                "question_id": qid, "repeat_index": rep, "condition": CONDITIONS[name],
                "target_model": cfg["model"], "question": prompt, "answer": result["answer"],
                "target_status": row["status"], "target_finish_reason": result.get("finish_reason"),
                "target_error": result.get("error_message"),
                "depth1": values[1], "depth2": values[2], "domain": values[3], "context_group": values[5],
                "target_condition_run_id": meta["run_id"], "target_attempt_id": result["attempt_id"],
                "target_error_type": result.get("error_type"), "target_error_code": result.get("error_code"),
                "target_http_status": result.get("http_status"),
            }
        if count != parent["questions_per_condition"] * cfg["repetitions"] or count != meta["planned_repetitions"]:
            raise ValueError(f"{name}: incomplete target rows ({count})")


def write_input(root, dest, filename):
    parent = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if parent.get("schema_version") != 3 or not parent.get("ready") or parent["condition_order"] != list(CONDITIONS):
        raise ValueError("Expected a prepared schema-version-3 six-condition target run")
    if not parent.get("run_id") or parent["questions_per_condition"] != parent["config"]["expected_rows"]:
        raise ValueError("Invalid target run ID or expected question count")
    path = dest / filename
    baseline = {}
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for name in CONDITIONS:
            ids = set()
            for row in read_condition(root, parent, name):
                identity = row["question_id"], row["repeat_index"]
                metadata = [row["question"], *[row[k] for k in STRATA]]
                if name == "c0":
                    baseline[identity] = metadata
                if identity in ids or metadata != baseline.get(identity):
                    raise ValueError(f"{name}/{identity}: duplicate or unmatched item")
                if not 1 <= identity[1] <= parent["config"]["repetitions"]:
                    raise ValueError(f"{name}/{identity}: invalid repeat index")
                ids.add(identity)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                n += 1
            if ids != baseline.keys():
                raise ValueError(f"{name}: question coverage differs from C0")
    if n != parent["planned_results"] or not n:
        raise ValueError(f"Expected {parent['planned_results']} rows; found {n}")
    return {"run_id": parent["run_id"], "target_model": parent["config"]["model"],
            "conditions": list(CONDITIONS.values()), "target_metadata": True, "files": [{"path": path.name}]}, path


