#!/usr/bin/env python3
"""Together judge implementation. Python 3.10+; no hidden SDK retries or live test mode.

All model inputs are existing question/answer pairs. This runner never asks a
target model to generate new answers. API failures are records, not refusals.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import contextmanager
from copy import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid

import openpyxl
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

VERSION = "1.4.2"
EXECUTION_VERSION = "1.4.2"
LABELS = ("compliant", "partially_compliant", "refusal", "clarification_request", "undetermined")
ROOT = Path(__file__).resolve().parent
SENSITIVE_HEADERS = {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_env(path):
    # Minimal .env reader: never executes text or prints the API key.
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "TOGETHER_API_KEY":
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    f = (directory / ".run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0, 2)
            if f.tell() == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError("Another runner is using this directory. Stop it or choose a new run directory.")
    try:
        yield
    finally:
        f.close()  # OS releases lock even after an unclean process exit.


def load_spec(config_path, input_override=None):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base = config_path.parent
    input_path = Path(input_override).resolve() if input_override else (base / cfg["input_file"]).resolve()
    prompt = (base / cfg["rubric_file"]).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Judge prompt is empty.")
    models = cfg["models"]
    if not models or len({m["alias"] for m in models}) != len(models):
        raise ValueError("Model alias is empty or duplicated.")
    if len({m["model_id"] for m in models}) != len(models):
        raise ValueError("Judge model IDs must be distinct.")
    protected = {"model", "messages", "tools", "tool_choice", "n"}
    for params in [cfg["common_parameters"], *[m.get("parameters", {}) for m in models]]:
        if protected.intersection(params):
            raise ValueError("parameters cannot override model, messages, tools or n.")
        if params.get("stream") is True:
            raise ValueError("This runner requires stream=false for complete response logging.")
    if cfg["checkpoint_every"] < 1 or cfg["timeout_seconds"] <= 0 or cfg["pause_seconds"] < 0:
        raise ValueError("Check the checkpoint, timeout and pause settings.")
    if not cfg["base_url"].startswith("https://"):
        raise ValueError("base_url must use HTTPS.")
    book = openpyxl.load_workbook(input_path, read_only=True, data_only=False)
    try:
        sheet = book[cfg["sheet_name"]]
        header = [c.value for c in next(sheet.iter_rows(min_row=1, max_row=1))]
        required = [cfg["id_column"], cfg["question_column"], *cfg["response_columns"].values()]
        for name in required:
            if header.count(name) != 1:
                raise ValueError(f"Required column missing or duplicated: {name}")
        cols = {name: header.index(name) for name in required}
        rows = []
        seen = set()
        for cells in sheet.iter_rows(min_row=2):
            vals = [c.value for c in cells]
            if not any(v is not None for v in vals):
                continue
            qid = vals[cols[cfg["id_column"]]]
            if not isinstance(qid, str) or not qid.strip() or qid != qid.strip() or qid in seen:
                raise ValueError(f"Invalid or duplicate question_id at Excel row {cells[0].row}.")
            seen.add(qid)
            question = vals[cols[cfg["question_column"]]]
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question for question_id: {qid}")
            answers = {}
            for condition, name in cfg["response_columns"].items():
                cell = cells[cols[name]]
                if cell.data_type == "f":
                    raise ValueError(f"Formulas in response columns are unsupported: {qid}/{condition}")
                if cell.value is not None and not isinstance(cell.value, str):
                    raise ValueError(f"Response must be a string: {qid}/{condition}")
                answers[condition] = cell.value
            if cells[cols[cfg["question_column"]]].data_type == "f":
                raise ValueError(f"Formula found in the question column: {qid}")
            rows.append({"question_id": qid, "excel_row": cells[0].row, "question": question, "answers": answers})
        if not rows:
            raise ValueError("No input questions found.")
    finally:
        book.close()
    cfg_for_hash = dict(cfg)
    cfg_for_hash.pop("input_file", None)
    cfg_for_hash.pop("rubric_file", None)
    signature = {"runner_version": EXECUTION_VERSION, "config": cfg_for_hash,
                 "input_sha256": digest(input_path.read_bytes()), "rubric_sha256": digest(prompt)}
    return {"config": cfg, "input_path": input_path, "prompt": prompt, "rows": rows,
            "fingerprint": digest(dumps(signature)), "signature": signature}


def job_meta(row, condition, model, phase="judge"):
    answer = row["answers"][condition]
    meta = {"phase": phase, "question_id": row["question_id"], "condition": condition,
            "judge_alias": model["alias"], "requested_model": model["model_id"],
            "input_excel_row": row["excel_row"], "question_sha256": digest(row["question"]),
            "response_sha256": digest(answer or "")}
    meta["job_key"] = digest(dumps([phase, meta["question_id"], condition, model["alias"]]))
    return meta


def make_payload(spec, model, question, answer):
    payload = dict(spec["config"]["common_parameters"])
    payload.update(model.get("parameters", {}))
    response_format = payload.get("response_format", {})
    if response_format.get("type") == "json_schema":
        schema = response_format["json_schema"]["schema"]
        system_prompt = spec["prompt"] + "\n\nReturn only JSON matching this JSON Schema:\n" + dumps(schema)
    elif response_format == {"type": "json_object"}:
        # JSON-object mode: send the rubric as is, without an appended schema.
        system_prompt = spec["prompt"]
    else:
        raise ValueError("response_format must be json_object or json_schema.")
    payload.update({"model": model["model_id"], "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({"original_question": question, "assistant_response": answer}, ensure_ascii=False)}
    ]})
    return payload


def transport_once(url, body, api_key, timeout, client_id):
    """Exactly one HTTP attempt. HTTPError retains its complete error body."""
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "Accept": "application/json", "User-Agent": f"LLM-Evaluation-Judge/{VERSION}",
        "X-Client-Request-Id": client_id,
    })
    # No redirects: never forward a credential to an unexpected host.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


class Audit:
    def __init__(self, run_dir, spec):
        self.directory = Path(run_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logs = self.directory / "logs"
        self.logs.mkdir(exist_ok=True)
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if self.manifest["fingerprint"] != spec["fingerprint"]:
                raise ValueError("Input, rubric or settings differ from the existing run. Use a new run directory.")
        else:
            snapshot = self.directory / "input_snapshot.xlsx"
            shutil.copyfile(spec["input_path"], snapshot)
            (self.directory / "judge_prompt_snapshot.txt").write_text(spec["prompt"], encoding="utf-8")
            self.manifest = {"run_id": str(uuid.uuid4()), "created_at": now(),
                "fingerprint": spec["fingerprint"], **spec["signature"],
                "original_input_path": str(spec["input_path"]), "row_count": len(spec["rows"]),
                "python_version": sys.version, "openpyxl_version": importlib.metadata.version("openpyxl")}
            atomic_json(manifest_path, self.manifest)
        snapshot = self.directory / "input_snapshot.xlsx"
        if not snapshot.exists() or digest(snapshot.read_bytes()) != self.manifest["input_sha256"]:
            raise ValueError("The run input snapshot changed or is missing.")
        self.db = sqlite3.connect(self.directory / "state.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS results(job_key TEXT PRIMARY KEY, result_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts(
                attempt_id TEXT PRIMARY KEY, job_key TEXT NOT NULL,
                meta_json TEXT NOT NULL, request_json TEXT NOT NULL,
                received_json TEXT, result_json TEXT);
        """)
        self.db.commit()

    def append(self, name, value):
        # SQLite is authoritative. JSONL is an independently readable audit log.
        path = self.logs / name
        with path.open("ab") as f:
            f.write((dumps(value) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())

    def get(self, key):
        row = self.db.execute("SELECT result_json FROM results WHERE job_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def results(self, phase="judge"):
        values = [json.loads(r[0]) for r in self.db.execute("SELECT result_json FROM results")]
        return [r for r in values if phase is None or r["phase"] == phase]

    def begin(self, meta, payload, url):
        client_id = str(uuid.uuid4())
        count = self.db.execute("SELECT COUNT(*) FROM attempts WHERE job_key=?", (meta["job_key"],)).fetchone()[0] + 1
        metadata = {**meta, "run_id": self.manifest["run_id"], "attempt_id": client_id,
                    "attempt_number": count, "started_at": now()}
        request = {**metadata, "method": "POST", "url": url,
            "headers": {"Authorization": "[REDACTED]", "Content-Type": "application/json",
                        "Accept": "application/json", "User-Agent": f"LLM-Evaluation-Judge/{VERSION}",
                        "X-Client-Request-Id": client_id},
            "body": payload, "body_utf8": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
        self.db.execute("INSERT INTO attempts VALUES(?,?,?,?,NULL,NULL)",
                        (client_id, meta["job_key"], dumps(metadata), dumps(request)))
        self.db.commit()
        self.append("requests.jsonl", request)
        return metadata, request["body_utf8"].encode("utf-8")

    def received(self, meta, response):
        self.db.execute("UPDATE attempts SET received_json=? WHERE attempt_id=?", (dumps(response), meta["attempt_id"]))
        self.db.commit()
        self.append("responses.jsonl", {**meta, **response})

    def record(self, result):
        result = {"run_id": self.manifest["run_id"], "finished_at": now(), **result}
        value = dumps(result)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO results VALUES(?,?)", (result["job_key"], value))
            if result.get("attempt_id"):
                self.db.execute("UPDATE attempts SET result_json=? WHERE attempt_id=?", (value, result["attempt_id"]))
        self.append("results.jsonl", result)
        if result["status"] not in ("OK", "INPUT_EMPTY_RESPONSE"):
            self.append("errors.jsonl", result)
        return result

    def recover(self):
        # Do not re-charge for a request that may have reached the provider.
        for row in list(self.db.execute("SELECT * FROM attempts WHERE result_json IS NULL ORDER BY rowid")):
            meta = json.loads(row["meta_json"])
            if row["received_json"]:
                result = interpret(meta, json.loads(row["received_json"]))
            else:
                result = error_result(meta, "INTERRUPTED_UNKNOWN", "Request completion was unknown at interruption. An explicit retry is required.")
            result["recovered"] = True
            self.record(result)

    def close(self):
        self.db.close()


def error_result(meta, status, message, **extra):
    return {**meta, "status": status, "label": "error", "error_message": f"[{status}] {message}", **extra}


def parse_label(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Final judge content is empty.")
    text = text.strip()
    # A few providers expose thought tags even in non-thinking mode. Preserve
    # the raw response in logs and parse only an explicit final-answer suffix.
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise ValueError("Unclosed think tag with no final answer.")
        text = text[end + len("</think>"):].strip()
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1)
    def no_duplicates(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj
    obj = json.loads(text, object_pairs_hook=no_duplicates)
    if not isinstance(obj, dict) or set(obj) != {"label"}:
        raise ValueError("Output must be a JSON object containing only the label key.")
    if obj["label"] not in LABELS:
        raise ValueError("The judge returned an unsupported label.")
    return obj


def interpret(meta, response):
    extras = {k: response.get(k) for k in ("http_status", "latency_ms", "received_at", "server_request_id")}
    if response.get("network_error"):
        return error_result(meta, response.get("error_status", "NETWORK_ERROR"), response["network_error"], **extras)
    status = response.get("http_status")
    raw = response.get("raw_body_text", "")
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        data = None
    if status is None or not 200 <= status < 300:
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            message = str(err.get("message", err))
            extras["provider_error_code"] = err.get("code")
            extras["provider_error_type"] = err.get("type")
        else:
            message = str(err or raw or "No HTTP response body.")
        return error_result(meta, "HTTP_ERROR", f"HTTP {status}: {message}", **extras)
    if not isinstance(data, dict):
        return error_result(meta, "INVALID_API_RESPONSE", "The API response body must be a JSON object.", **extras)
    extras.update({"returned_model": data.get("model"), "response_id": data.get("id"), "usage": data.get("usage") or {}})
    try:
        choice = data["choices"][0]
        message = choice["message"]
        if not isinstance(choice, dict) or not isinstance(message, dict):
            return error_result(meta, "INVALID_API_RESPONSE", "Each choice and its message must be JSON objects.", **extras)
        finish = choice.get("finish_reason")
        extras["finish_reason"] = finish
        if finish == "length":
            return error_result(meta, "JUDGE_TRUNCATED", "Judge output reached the token limit and was excluded.", **extras)
        if finish not in ("stop", "eos", "eos_token"):
            return error_result(meta, "JUDGE_INCOMPLETE", f"Incomplete generation; finish_reason: {finish}", **extras)
        if message.get("refusal"):
            return error_result(meta, "JUDGE_REFUSAL", "The judge declined to evaluate the response.", **extras)
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return error_result(meta, "JUDGE_EMPTY_OUTPUT", "Final judge content is empty.", **extras)
        parsed = parse_label(content)
        if meta["phase"] == "smoke" and parsed["label"] != "compliant":
            return error_result(meta, "SMOKE_UNEXPECTED_LABEL", "Expected compliant for the 2+2/4 smoke pair.", **extras)
        return {**meta, **extras, "status": "OK", **parsed}
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        return error_result(meta, "INVALID_JUDGE_OUTPUT", str(exc), **extras)


def perform_attempt(audit, spec, meta, payload, api_key, transport=transport_once):
    url = spec["config"]["base_url"].rstrip("/") + "/chat/completions"
    meta, body = audit.begin(meta, payload, url)
    started = time.monotonic()
    interrupted = False
    try:
        status, headers, raw = transport(url, body, api_key, spec["config"]["timeout_seconds"], meta["attempt_id"])
        clean_headers = {k: ("[REDACTED]" if k.lower() in SENSITIVE_HEADERS else v) for k, v in headers.items()}
        lower = {k.lower(): v for k, v in headers.items()}
        response = {"http_status": status, "headers": clean_headers,
            "raw_body_text": raw.decode("utf-8", errors="replace"), "raw_body_base64": base64.b64encode(raw).decode("ascii"),
            "server_request_id": next((lower[k] for k in ("x-request-id", "x-together-request-id", "request-id", "x-amzn-requestid") if k in lower), None)}
    except KeyboardInterrupt:
        interrupted = True
        response = {"http_status": None, "network_error": "Interrupted by the user. Server completion is unknown.", "error_status": "INTERRUPTED_UNKNOWN"}
    except Exception as exc:
        response = {"http_status": None, "network_error": f"{type(exc).__name__}: {exc}", "error_status": "NETWORK_ERROR"}
    response.update({"received_at": now(), "latency_ms": round((time.monotonic() - started) * 1000, 3)})
    audit.received(meta, response)
    result = audit.record(interpret(meta, response))
    if interrupted:
        raise KeyboardInterrupt
    return result


def ensure_smoke(audit, spec, api_key, retry_errors=False, transport=transport_once):
    readiness = {}
    for model in spec["config"]["models"]:
        row = {"question_id": "__SMOKE__", "excel_row": 0, "question": "What is 2 + 2?", "answers": {"SMOKE": "4."}}
        meta = job_meta(row, "SMOKE", model, phase="smoke")
        prior = audit.get(meta["job_key"])
        if prior is None or (retry_errors and prior["status"] != "OK"):
            prior = perform_attempt(audit, spec, meta, make_payload(spec, model, row["question"], "4."), api_key, transport)
        readiness[model["alias"]] = prior
        print(f"[smoke] {model['alias']}: {prior['status']}", flush=True)
    return readiness


def safe_cell(sheet, row, col, value):
    cell = sheet.cell(row, col)
    if isinstance(value, (dict, list)):
        value = dumps(value)
    if isinstance(value, str):
        # Raw text always remains intact in SQLite/JSONL. Excel has a 32767-char limit.
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
        if len(value) > 32767:
            value = value[:32710] + " [See logs/results.jsonl for the complete text]"
        cell.value = value
        cell.data_type = "s"  # Never execute model text as an Excel formula.
    else:
        cell.value = value
    return cell


def export_workbook(audit, spec):
    """User-side Excel export; keep source cells and question_id order intact."""
    book = openpyxl.load_workbook(audit.directory / "input_snapshot.xlsx")
    cfg = spec["config"]
    sheet = book[cfg["sheet_name"]]
    headers = [c.value for c in sheet[1]]
    original_col = {name: headers.index(name) + 1 for name in cfg["response_columns"].values()}
    model_count = len(cfg["models"])
    count = model_count
    old_style = copy(next(iter(sheet.tables.values())).tableStyleInfo) if sheet.tables else None
    for name in list(sheet.tables):
        del sheet.tables[name]
    # Insert from right to left so original physical column positions remain valid.
    for position in sorted(original_col.values(), reverse=True):
        sheet.insert_cols(position + 1, count)
    mapping = {}
    headfill = PatternFill("solid", fgColor="243B53")
    for condition, source_name in cfg["response_columns"].items():
        position = next(c.column for c in sheet[1] if c.value == source_name)
        sheet.column_dimensions[get_column_letter(position)].width = 90
        for i, model in enumerate(cfg["models"]):
            label_col = position + 1 + i
            mapping[(condition, model["alias"])] = label_col
            cell = safe_cell(sheet, 1, label_col, f"{condition} {model['display_name']} compliance_label")
            cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
            cell.fill = headfill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            sheet.column_dimensions[get_column_letter(label_col)].width = 24
    sheet.row_dimensions[1].height = 45
    sheet.column_dimensions["A"].width = 16
    sheet.column_dimensions["B"].width = 60
    id_rows = {row["question_id"]: row["excel_row"] for row in spec["rows"]}
    records = audit.results()
    for result in records:
        r = id_rows[result["question_id"]]
        label_col = mapping[(result["condition"], result["judge_alias"])]
        cell = safe_cell(sheet, r, label_col, result["label"])
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        cell.font = Font(name="Arial", size=11)
    # Existing blank-response formatting must move with its original columns.
    sheet.conditional_formatting = openpyxl.formatting.formatting.ConditionalFormattingList()
    for condition, name in cfg["response_columns"].items():
        pos = next(c.column for c in sheet[1] if c.value == name)
        if condition in ("C3_E", "C3_P"):
            letter = get_column_letter(pos)
            sheet.conditional_formatting.add(f"{letter}2:{letter}{sheet.max_row}", CellIsRule(operator="equal", formula=['""'], fill=PatternFill("solid", fgColor="FFF0D9")))
        for model in cfg["models"]:
            labelcol = get_column_letter(mapping[(condition, model["alias"])])
            sheet.conditional_formatting.add(f"{labelcol}2:{labelcol}{sheet.max_row}", CellIsRule(operator="equal", formula=['"error"'], fill=PatternFill("solid", fgColor="FDE7E7")))
    table = Table(displayName="SampleResponses", ref=f"A1:{get_column_letter(sheet.max_column)}{sheet.max_row}")
    table.tableStyleInfo = old_style or TableStyleInfo(name="TableStyleLight9", showRowStripes=True)
    sheet.add_table(table)
    sheet.freeze_panes = "C2"
    details = book.create_sheet("judge_details")
    fields = ["question_id", "condition", "judge_alias", "status", "label", "attempt_id",
              "attempt_number", "server_request_id", "response_id", "http_status", "provider_error_code",
              "requested_model", "returned_model", "finish_reason", "latency_ms", "usage", "finished_at"]
    details.append(fields)
    row_order = {r["question_id"]: i for i, r in enumerate(spec["rows"])}
    condition_order = {c: i for i, c in enumerate(cfg["response_columns"])}
    model_order = {m["alias"]: i for i, m in enumerate(cfg["models"])}
    records.sort(key=lambda r: (row_order[r["question_id"]], condition_order[r["condition"]], model_order[r["judge_alias"]]))
    for r, record in enumerate(records, 2):
        for c, name in enumerate(fields, 1):
            safe_cell(details, r, c, record.get(name))
    details.freeze_panes = "D2"
    details.auto_filter.ref = details.dimensions
    for col in range(1, len(fields) + 1):
        details.column_dimensions[get_column_letter(col)].width = 25
    summary = book.create_sheet("judge_summary")
    total = len(spec["rows"]) * len(cfg["response_columns"]) * model_count
    counts = Counter(r["status"] for r in records)
    info = [["item", "value"], ["run_id", audit.manifest["run_id"]], ["rubric_file", cfg["rubric_file"]],
            ["rubric_version", cfg["rubric_version"]], ["output_format_version", cfg["output_format_version"]],
            ["rubric_sha256", audit.manifest["rubric_sha256"]],
            ["input_sha256", audit.manifest["input_sha256"]], ["input_question_count", len(spec["rows"])],
            ["planned_judge_slots", total], ["processed_slots", len(records)], ["pending_slots", total - len(records)],
            ["status_counts", dict(counts)], ["common_parameters", cfg["common_parameters"]],
            ["API endpoint", cfg["base_url"]], ["automatic_retries", 0], ["exported_at_utc", now()],
            ["blank_label_meaning", "Empty source responses and pending tasks have blank labels. The judge_details sheet marks empty responses as INPUT_EMPTY_RESPONSE."],
            ["error_meaning", "Judge errors are recorded separately from target response content."],
            ["full_text_records", "Raw requests and responses are in logs/requests.jsonl and responses.jsonl. Labels, statuses and errors are in state.sqlite3 and logs/results.jsonl. Judges return a label only."],
            ["output_format", f"All judges share one output workbook. Each source response is followed by {model_count} label columns. Judges return only the label."],
            ["conditions", list(cfg["response_columns"])]]
    for model in cfg["models"]:
        info.append([model["alias"], model])
    for r, row in enumerate(info, 1):
        for c, value in enumerate(row, 1):
            safe_cell(summary, r, c, value)
    summary.column_dimensions["A"].width = 28
    summary.column_dimensions["B"].width = 110
    for ws in (details, summary):
        ws.sheet_view.showGridLines = False
        for cell in ws[1]:
            cell.fill = headfill
            cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
    # Exact preservation gate before saving, independent of physical column offsets.
    finalheaders = [c.value for c in sheet[1]]
    for row in spec["rows"]:
        r = row["excel_row"]
        assert sheet.cell(r, finalheaders.index(cfg["id_column"]) + 1).value == row["question_id"]
        assert sheet.cell(r, finalheaders.index(cfg["question_column"]) + 1).value == row["question"]
        for cond, name in cfg["response_columns"].items():
            assert sheet.cell(r, finalheaders.index(name) + 1).value == row["answers"][cond]
    target = audit.directory / "judge_results.xlsx"
    tmp = audit.directory / "judge_results.tmp.xlsx"
    book.save(tmp)
    book.close()
    try:
        os.replace(tmp, target)
    except PermissionError:
        # Excel may lock the workbook on Windows. Keep running without data loss.
        target = audit.directory / f"judge_results_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.xlsx"
        os.replace(tmp, target)
    audit.append("events.jsonl", {"event": "excel_export", "at": now(), "path": str(target), "processed_slots": len(records),
                                  "exporter_version": VERSION, "layout": "labels_only"})
    return target


def safe_export(audit, spec):
    try:
        path = export_workbook(audit, spec)
        print(f"[saved] {path}", flush=True)
        return path
    except OSError as exc:
        audit.append("events.jsonl", {"event": "excel_export_failed", "at": now(), "error": str(exc)})
        print(f"[export warning] Excel export failed; database and JSONL records are saved: {exc}", file=sys.stderr, flush=True)
        return None


def run_jobs(audit, spec, api_key, retry_errors=False, max_calls=None, transport=transport_once):
    ready = ensure_smoke(audit, spec, api_key, retry_errors, transport)
    actual_calls = processed = 0
    completed = len(audit.results())
    total = len(spec["rows"]) * len(spec["config"]["response_columns"]) * len(spec["config"]["models"])
    for row in spec["rows"]:
        for condition, answer in row["answers"].items():
            for model in spec["config"]["models"]:
                meta = job_meta(row, condition, model)
                previous = audit.get(meta["job_key"])
                if previous and (not retry_errors or previous["status"] in ("OK", "INPUT_EMPTY_RESPONSE")):
                    continue
                if answer is None or not answer.strip():
                    result = audit.record({**meta, "status": "INPUT_EMPTY_RESPONSE", "label": None,
                        "skip_reason": "Empty target response; the judge API was skipped. See INPUT_EMPTY_RESPONSE records."})
                elif ready[model["alias"]]["status"] != "OK":
                    smoke = ready[model["alias"]]
                    result = audit.record(error_result(meta, "MODEL_PREFLIGHT_FAILED",
                        f"Judge skipped after a failed smoke test: {smoke['status']}. preflight_attempt_id={smoke.get('attempt_id')}",
                        preflight_attempt_id=smoke.get("attempt_id")))
                else:
                    if max_calls is not None and actual_calls >= max_calls:
                        return actual_calls
                    result = perform_attempt(audit, spec, meta, make_payload(spec, model, row["question"], answer), api_key, transport)
                    actual_calls += 1
                    if spec["config"]["pause_seconds"]:
                        time.sleep(spec["config"]["pause_seconds"])
                processed += 1
                if previous is None:
                    completed += 1
                print(f"[{completed}/{total}] {row['question_id']} {condition} {model['alias']} {result['status']} {result['label'] or '-'}", flush=True)
                if processed % spec["config"]["checkpoint_every"] == 0:
                    safe_export(audit, spec)
    return actual_calls


def describe(spec):
    counts = {cond: sum(bool(r["answers"][cond] and r["answers"][cond].strip()) for r in spec["rows"]) for cond in spec["config"]["response_columns"]}
    models = len(spec["config"]["models"])
    return {"questions": len(spec["rows"]), "nonblank_responses_by_condition": counts,
        "judge_models": [m["model_id"] for m in spec["config"]["models"]],
        "planned_judge_calls": sum(counts.values()) * models,
        "input_empty_slots": (len(spec["rows"]) * len(counts) - sum(counts.values())) * models,
        "initial_smoke_calls": models, "fingerprint": spec["fingerprint"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "run", "export"])
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--input", help="Override the input file; changed inputs require a new run directory.")
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "example"))
    parser.add_argument("--max-calls", type=int, help="Limit evaluation calls in this invocation; smoke calls are counted separately.")
    parser.add_argument("--retry-errors", action="store_true", help="Explicitly retry recorded errors; additional API charges may apply.")
    args = parser.parse_args(argv)
    if args.max_calls is not None and args.max_calls < 1:
        parser.error("--max-calls must be positive.")
    if args.max_calls is not None and args.command != "run":
        parser.error("--max-calls is available only with run.")
    audit = None
    try:
        spec = load_spec(args.config, args.input)
        print(json.dumps(describe(spec), ensure_ascii=False, indent=2), flush=True)
        directory = Path(args.run_dir).resolve()
        with run_lock(directory):
            audit = Audit(directory, spec)
            audit.recover()
            if args.command == "prepare":
                atomic_json(directory / "plan.json", describe(spec))
                print("Prepared with zero API calls. run performs one smoke test per model before evaluation.")
            elif args.command == "export":
                if safe_export(audit, spec) is None:
                    return 2
            else:
                load_env(Path(args.config).resolve().parent / ".env")
                key = os.environ.get("TOGETHER_API_KEY", "").strip()
                if not key or "YOUR-" in key:
                    raise ValueError("Set TOGETHER_API_KEY in the environment or .env file.")
                if args.command == "smoke":
                    ready = ensure_smoke(audit, spec, key, args.retry_errors)
                    return 0 if all(r["status"] == "OK" for r in ready.values()) else 3
                else:
                    try:
                        calls = run_jobs(audit, spec, key, args.retry_errors, args.max_calls)
                        print(f"Evaluation HTTP calls in this invocation: {calls}; automatic retries disabled.")
                    finally:
                        output = safe_export(audit, spec)
                    counts = Counter(r["status"] for r in audit.results())
                    print("Final status:", dumps(dict(counts)), flush=True)
                    return 0 if output else 2
    except KeyboardInterrupt:
        print("\nInterrupted. Repeat the same command to continue from saved state.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as exc:
        print(f"[setup/storage error] {exc}", file=sys.stderr)
        return 2
    finally:
        if audit:
            audit.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
