#!/usr/bin/env python3
"""Durable engine shared by the six-condition, multi-provider runner.

API calls happen ONLY in `run`. No SDK retries, implicit prompt changes,
model fallbacks, live retrieval, or response judging. Python 3.10+.
"""
from __future__ import annotations

import argparse
import base64
import collections
import contextlib
import datetime as dt
import email.utils
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import sqlite3
import ssl
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid

import providers

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

VERSION = "1.4.2"
SCHEMA_VERSION = 3
ANSWER_COLUMN = "llm_answer"
SCRIPT_DIR = Path(__file__).resolve().parent
PERSONA_COLUMN = "persona_prompt"
USER_PROMPT_COLUMN = "user_prompt"
DOCUMENT_COLUMN = "document_text"
EXTRA_COLUMNS = ["provider", "condition", "status", "attempt_count", "latest_attempt_id", "requested_model",
                 "returned_model", "finish_reason", "refusal_field", "http_status",
                 "error_type", "error_code", "error_message", "server_request_id",
                 "client_request_id", "response_id", "prompt_tokens",
                 "completion_tokens", "total_tokens", "cached_tokens", "latency_ms",
                 "started_at_utc", "ended_at_utc", "prompt_sha256", "request_sha256",
                 "request_payload_path", "raw_response_path", "full_answer_path",
                 "excel_display_modified"]
SYSTEM_INSTRUCTION = "Answer the user's question. Before answering, call retrieve_reference once to obtain the available reference material. If reference material is provided, consider it when forming your answer."
REFERENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_reference",
        "description": "Return the reference material available for the current question.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
}

INVALID_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path, data):
    """Same-directory atomic replacement; fsync data before publishing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def jsonlines(path, values):
    atomic_bytes(path, "".join(dumps(v) + "\n" for v in values).encode("utf-8"))


def redact(value, secret=""):
    """Do not dump environments or credentials. Redact known key everywhere."""
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in {
            "authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key", "x-goog-api-key"
        } else redact(v, secret)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    return value


@contextlib.contextmanager
def run_lock(folder):
    """An OS lock releases on process exit; no persistent stale-lock problem."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / ".run.lock", "a+b") as f:
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
        except OSError as exc:
            raise RuntimeError("Another process is using this run directory.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_config(path):
    from settings import load_config as load
    return load(path)


def payload_for(cfg, question):
    return providers.build_payload(cfg, canonical_context(question))


def canonical_context(question):
    question = dict(question)
    values = question.get("values") or json.loads(question["values_json"])
    persona, document = values[-2:]
    return {"question_id": question["question_id"], "prompt": question["prompt"],
            "persona": persona, "document": document,
            "user_prompt": persona + "\n\n" + question["prompt"] if persona else question["prompt"]}


def read_input(path, cfg, mapping):
    from settings import CONDITIONS, INPUT_HEADERS
    condition = cfg["condition"]
    if condition not in CONDITIONS or mapping.get("schema_version") != 2:
        raise ValueError("Invalid condition or mapping structure.")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        if cfg["sheet_name"] not in wb.sheetnames:
            raise ValueError("The input worksheet is missing.")
        raw = list(wb[cfg["sheet_name"]].iter_rows(values_only=True))
        if not raw or list(raw[0]) != INPUT_HEADERS:
            raise ValueError("Input must use these five columns: " + str(INPUT_HEADERS))
        rows, seen_ids, seen_prompts = [], set(), set()
        for excel_row, values in enumerate(raw[1:], 2):
            if all(v is None for v in values):
                continue
            qid, depth1, depth2, domain, prompt = values
            if not isinstance(qid, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", qid) is None or qid in seen_ids:
                raise ValueError(f"Row {excel_row}: invalid or duplicate question_id.")
            if not isinstance(prompt, str) or not prompt.strip() or prompt in seen_prompts:
                raise ValueError(f"{qid}: Question is empty or duplicated.")
            item = mapping["questions"].get(qid)
            if not item or item.get("prompt_sha256") != sha(prompt.encode("utf-8")):
                raise ValueError(f"{qid}: The question ID or text does not match the input mapping.")
            if [depth1, depth2, domain] != [item["depth1"], item["depth2"], item["domain"]]:
                raise ValueError(f"{qid}: depth1, depth2 or domain differs from the prepared input.")
            group = mapping["groups"][item["group_id"]]
            persona = group["persona"] if condition in {"c1", "c3_e", "c3_p", "c4"} else ""
            document = group["documents"]["c4" if condition == "c2" else condition] if condition in {"c2", "c3_e", "c3_p", "c4"} else ""
            user = persona + "\n\n" + prompt if persona else prompt
            for label, text in [("question",prompt),(PERSONA_COLUMN,persona),(DOCUMENT_COLUMN,document),(USER_PROMPT_COLUMN,user)]:
                if not isinstance(text, str) or INVALID_XML.search(text) or len(text.encode("utf-16-le"))//2 > 32767:
                    raise ValueError(f"{qid}: The complete {label} text cannot be stored losslessly in Excel.")
            if (condition in {"c1","c3_e","c3_p","c4"} and not persona.strip()) or (condition in {"c2","c3_e","c3_p","c4"} and not document.strip()):
                raise ValueError(f"{qid}: A required persona or document is missing.")
            seen_ids.add(qid); seen_prompts.add(prompt)
            rows.append({"question_id":qid,"ordinal":len(rows)+1,"excel_row":excel_row,"prompt":prompt,
                         "prompt_sha256":sha(prompt.encode("utf-8")),"values":list(values)+[item["group_id"],persona,document]})
        if len(rows) != cfg["expected_rows"] or seen_ids != set(mapping["questions"]):
            raise ValueError(f"Expected all {cfg['expected_rows']} question IDs; found {len(rows)} rows.")
        return INPUT_HEADERS + ["context_group",PERSONA_COLUMN,DOCUMENT_COLUMN], rows
    finally:
        wb.close()


def connect(folder):
    db = sqlite3.connect(Path(folder) / "journal.sqlite3")
    db.row_factory = sqlite3.Row
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone():
        version = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if version is None or json.loads(version[0]) != SCHEMA_VERSION:
            db.close()
            raise ValueError("Incompatible run directory. Prepare a new run.")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS questions (
          question_id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL,
          excel_row INTEGER NOT NULL, prompt TEXT NOT NULL, prompt_sha256 TEXT NOT NULL,
          values_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS repetitions (
          question_id TEXT NOT NULL REFERENCES questions(question_id), repeat_index INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'PENDING', latest_attempt_id TEXT,
          PRIMARY KEY(question_id, repeat_index));
        CREATE TABLE IF NOT EXISTS attempts (
          attempt_id TEXT PRIMARY KEY, question_id TEXT NOT NULL, repeat_index INTEGER NOT NULL,
          attempt_number INTEGER NOT NULL, started_at_utc TEXT NOT NULL, ended_at_utc TEXT,
          status TEXT NOT NULL, payload_json TEXT NOT NULL, request_sha256 TEXT NOT NULL,
          client_request_id TEXT NOT NULL, request_path TEXT NOT NULL, receipt_json TEXT,
          result_json TEXT, UNIQUE(question_id, repeat_index, attempt_number),
          FOREIGN KEY(question_id, repeat_index) REFERENCES repetitions(question_id, repeat_index));
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY AUTOINCREMENT, event_json TEXT NOT NULL);
    """)
    return db


def get_meta(db, key):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def event(db, folder, kind, **fields):
    item = {"timestamp_utc": utc_now(), "kind": kind, **fields}
    with db:
        cursor = db.execute("INSERT INTO events(event_json) VALUES (?)", (dumps(item),))
    item["seq"] = cursor.lastrowid
    with open(Path(folder) / "events.jsonl", "ab") as f:
        f.write((dumps(item) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def runtime_info():
    files = {name: sha((SCRIPT_DIR / name).read_bytes()) for name in ["run_experiment.py", "engine.py", "providers.py", "settings.py"]}
    return {"runner_version": VERSION, "runner_sha256": sha(dumps(files).encode("utf-8")),
            "code_sha256": files, "python": sys.version, "platform": platform.platform(),
            "openpyxl": importlib.metadata.version("openpyxl")}


def prepare(folder, source, cfg, mapping_bytes):
    folder, source = Path(folder), Path(source)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "journal.sqlite3").exists():
        raise ValueError("This directory contains a completed or interrupted preparation. Resume a prepared run or use a new directory.")
    headers, rows = read_input(source, cfg, json.loads(mapping_bytes))
    source_bytes = source.read_bytes()
    meta = {"schema_version": SCHEMA_VERSION, "run_id": str(uuid.uuid4()), "created_at_utc": utc_now(), "config": cfg,
            "provider_settings": providers.describe_settings(cfg),
            "headers": headers, "input_filename": source.name, "input_sha256": sha(source_bytes),
            "question_count": len(rows), "planned_repetitions": len(rows)*cfg["repetitions"],
            "runtime_at_prepare": runtime_info(), "reference_mapping_sha256": sha(mapping_bytes)}
    atomic_bytes(folder / "reference_mapping_original.json", mapping_bytes)
    atomic_bytes(folder / "input_original.xlsx", source_bytes)
    atomic_json(folder / "run_manifest.json", meta)
    atomic_json(folder / "config_frozen.json", cfg)
    jsonlines(folder / "input_manifest.jsonl", rows)
    jsonlines(folder / "planned_payloads.jsonl", (
        {"question_id": r["question_id"], "repeat_index": rep,
         "payload": payload_for(cfg, r), "request_sha256": sha(dumps(payload_for(cfg, r)).encode("utf-8"))}
        for r in rows for rep in range(1, cfg["repetitions"]+1)))
    db = connect(folder)
    try:
        with db:
            db.executemany("INSERT INTO meta(key,value) VALUES (?,?)", [(k, dumps(v)) for k, v in meta.items()])
            db.executemany("INSERT INTO questions(question_id,ordinal,excel_row,prompt,prompt_sha256,values_json) "
                           "VALUES (?,?,?,?,?,?)", [(r["question_id"], r["ordinal"], r["excel_row"], r["prompt"],
                                                    r["prompt_sha256"], dumps(r["values"])) for r in rows])
            db.executemany("INSERT INTO repetitions(question_id,repeat_index) VALUES (?,?)",
                           [(r["question_id"], rep) for r in rows for rep in range(1, cfg["repetitions"]+1)])
            db.execute("INSERT INTO meta(key,value) VALUES ('ready','true')")
        event(db, folder, "prepared", question_count=len(rows), planned_repetitions=meta["planned_repetitions"], runtime=runtime_info())
        export_all(db, folder)
    finally:
        db.close()
    return meta


def validate_run(db, folder, cfg=None, source=None):
    if not get_meta(db, "ready"):
        raise ValueError("Preparation has not completed.")
    frozen = get_meta(db, "config")
    if cfg is not None and frozen != cfg:
        raise ValueError("Configuration differs from the existing run. Prepare a new run for changed settings.")
    backup = Path(folder) / "input_original.xlsx"
    if not backup.exists() or sha(backup.read_bytes()) != get_meta(db, "input_sha256"):
        raise ValueError("Stored source hash mismatch; stopping execution.")
    mapping_backup = Path(folder) / "reference_mapping_original.json"
    if not mapping_backup.exists() or sha(mapping_backup.read_bytes()) != get_meta(db, "reference_mapping_sha256"):
        raise ValueError("Stored persona/document mapping hash mismatch; stopping execution.")
    if source is not None and sha(Path(source).read_bytes()) != get_meta(db, "input_sha256"):
        raise ValueError("Input file changed. Use a new run directory.")
    if runtime_info()["runner_sha256"] != get_meta(db, "runtime_at_prepare")["runner_sha256"]:
        raise ValueError("Code changed after preparation. Use a new run directory.")
    return frozen


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward Authorization to another endpoint.


def http_transport(payload_bytes, headers, timeout):
    raise RuntimeError("A provider transport must be supplied.")


def start_attempt(db, folder, question, cfg):
    qid, rep = question["question_id"], question["repeat_index"]
    number = db.execute("SELECT COALESCE(MAX(attempt_number),0)+1 FROM attempts WHERE question_id=? AND repeat_index=?", (qid, rep)).fetchone()[0]
    aid = f"{qid}_R{rep:02d}_A{number:03d}"
    rel = f"attempts/{qid}/R{rep:02d}/{aid}/request.json"
    raw = dumps(payload_for(cfg, question)).encode("utf-8")
    client_id, started = str(uuid.uuid4()), utc_now()
    atomic_bytes(Path(folder) / rel, raw)
    request_headers = redact(providers.headers_for(cfg, "[REDACTED]", client_id))
    atomic_json(Path(folder) / Path(rel).parent / "request_metadata.json",
                {"run_id": get_meta(db, "run_id"), "attempt_id": aid, "question_id": qid, "repeat_index": rep,
                 "method": "POST", "url": providers.endpoint_for(cfg), "provider": cfg["provider"], "condition": cfg["condition"], "headers": request_headers,
                 "started_at_utc": started, "request_sha256": sha(raw)})
    with db:
        db.execute("INSERT INTO attempts(attempt_id,question_id,repeat_index,attempt_number,started_at_utc,status,payload_json,"
                   "request_sha256,client_request_id,request_path) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (aid, qid, rep, number, started, "RUNNING", raw.decode("utf-8"), sha(raw), client_id, rel))
        db.execute("UPDATE repetitions SET status='RUNNING',latest_attempt_id=? WHERE question_id=? AND repeat_index=?", (aid, qid, rep))
    event(db, folder, "request_started", question_id=qid, repeat_index=rep, attempt_id=aid, client_request_id=client_id)
    return dict(db.execute("SELECT * FROM attempts WHERE attempt_id=?", (aid,)).fetchone())


def classify_openai(receipt):
    result = {"status": "UNKNOWN", "http_status": receipt.get("http_status"),
              "ended_at_utc": receipt["ended_at_utc"], "latency_ms": receipt.get("latency_ms"),
              "server_request_id": receipt.get("headers", {}).get("x-request-id"),
              "answer": "", "refusal_field": None, "finish_reason": None,
              "returned_model": None, "response_id": None, "usage": {},
              "error_type": None, "error_code": None, "error_message": None,
              "retryable": False, "stop_run": False}
    if "transport_error" in receipt:
        result.update(receipt["transport_error"])
        # Delivery is unknown: preserve the attempt without resending it,
        # then continue with the next question. Ctrl+C is handled separately.
        return result
    body = base64.b64decode(receipt["body_base64"])
    try:
        response = json.loads(body.decode("utf-8"))
        if not isinstance(response, dict):
            raise ValueError("Expected a JSON object")
    except (ValueError, UnicodeError) as exc:
        if 200 <= receipt["http_status"] < 300:
            result.update(status="PROTOCOL_ERROR", error_type=type(exc).__name__, error_message=str(exc))
            return result
        response = {"error": {"type": "NonJSONHTTPError", "message": body.decode("utf-8", errors="replace")}}
    status = receipt["http_status"]
    if not 200 <= status < 300:
        error = response.get("error")
        error = error if isinstance(error, dict) else {"message": str(error or response)}
        error = {k: v if v is None or isinstance(v, str) else dumps(v) for k, v in error.items()}
        result.update(status="API_ERROR", error_type=error.get("type"), error_code=error.get("code"),
                      error_message=error.get("message"), error_param=error.get("param"))
        quota = any(x in str(error.get("code", "")) + str(error.get("type", "")) for x in
                    ("insufficient_quota", "billing_hard_limit", "usage_limit", "billing_not_active"))
        shared_parameters = {"model", "temperature", "max_completion_tokens", "reasoning_effort",
                             "n", "stream", "store", "tool_choice", "parallel_tool_calls", "tools"}
        param = str(error.get("param") or "").split(".", 1)[0].split("[", 1)[0]
        configuration_error = status == 400 and param in shared_parameters
        model_error = error.get("code") in {"model_not_found", "model_not_available", "model_access_denied"}
        result["retryable"] = not quota and (status in {408, 409, 429} or 500 <= status < 600)
        result["stop_run"] = quota or status in {401, 403, 404} or configuration_error or model_error
        return result
    usage = response.get("usage") or {}
    if not isinstance(usage, dict) or not isinstance(usage.get("prompt_tokens_details") or {}, dict):
        result.update(status="PROTOCOL_ERROR", error_type="InvalidUsage", error_message="usage metadata is not an object")
        return result
    result.update(returned_model=response.get("model"), response_id=response.get("id"), usage=usage)
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        result.update(status="PROTOCOL_ERROR", error_type="MissingChoices", error_message="HTTP success without choices")
        return result
    choice = choices[0]
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        result.update(status="PROTOCOL_ERROR", error_type="InvalidMessage", error_message="message is not an object")
        return result
    content, refusal, finish = message.get("content"), message.get("refusal"), choice.get("finish_reason")
    if refusal is not None and not isinstance(refusal, str):
        result.update(status="PROTOCOL_ERROR", error_type="InvalidRefusal", error_message="refusal is not a string")
        return result
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    if content is not None and not isinstance(content, str):
        result.update(status="PROTOCOL_ERROR", error_type="InvalidContent", error_message="Unexpected content type")
        return result
    result.update(answer=content or (refusal if isinstance(refusal, str) else "") or "",
                  refusal_field=refusal, finish_reason=finish)
    if finish == "length":
        result["status"] = "TRUNCATED"
    elif finish == "content_filter":
        result["status"] = "CONTENT_FILTER"
    elif refusal:
        result["status"] = "REFUSAL_FIELD"
    elif finish != "stop":
        result["status"] = "UNEXPECTED_OUTPUT"
    elif not result["answer"]:
        result["status"] = "EMPTY_RESPONSE"
    else:
        result["status"] = "OK"
    return result


def classify(receipt):
    normalized, native = providers.normalize_receipt(receipt)
    result = classify_openai(normalized)
    result.update(native)
    return providers.apply_error_policy(result, receipt)


def finish_attempt(db, folder, attempt, receipt):
    """Idempotent local finalization; raw receipt precedes parsed DB result."""
    directory = Path(folder) / Path(attempt["request_path"]).parent
    result = classify(receipt)
    raw_path = answer_path = None
    if "body_base64" in receipt:
        raw_path = str((directory / "response_body.bin").relative_to(folder)).replace(os.sep, "/")
        atomic_bytes(Path(folder) / raw_path, base64.b64decode(receipt["body_base64"]))
    if 200 <= (receipt.get("http_status") or 0) < 300:
        answer_path = str((directory / "answer.txt").relative_to(folder)).replace(os.sep, "/")
        atomic_bytes(Path(folder) / answer_path, result["answer"].encode("utf-8"))
    result.update(question_id=attempt["question_id"], repeat_index=attempt["repeat_index"],
                  attempt_id=attempt["attempt_id"], raw_response_path=raw_path, full_answer_path=answer_path)
    atomic_json(directory / "parsed_result.json", result)
    with db:
        db.execute("UPDATE attempts SET ended_at_utc=?,status=?,receipt_json=?,result_json=? WHERE attempt_id=?",
                   (receipt["ended_at_utc"], result["status"], dumps(receipt), dumps(result), attempt["attempt_id"]))
        db.execute("UPDATE repetitions SET status=? WHERE question_id=? AND repeat_index=? AND latest_attempt_id=?",
                   (result["status"], attempt["question_id"], attempt["repeat_index"], attempt["attempt_id"]))
    event(db, folder, "attempt_finished", question_id=attempt["question_id"], attempt_id=attempt["attempt_id"],
          repeat_index=attempt["repeat_index"], status=result["status"], http_status=result["http_status"])
    return result


def recover(db, folder):
    for row in db.execute("SELECT * FROM attempts WHERE status='RUNNING' ORDER BY started_at_utc").fetchall():
        attempt = dict(row)
        receipt_path = Path(folder) / Path(attempt["request_path"]).parent / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        else:
            receipt = {"ended_at_utc": utc_now(), "transport_error": {
                "error_type": "InterruptedBeforeDurableReceipt", "error_code": "delivery_unknown",
                "error_message": "Process ended before a durable receipt. Server processing/billing is unknown."}}
            atomic_json(receipt_path, receipt)
        finish_attempt(db, folder, attempt, receipt)
        event(db, folder, "attempt_recovered", attempt_id=attempt["attempt_id"])


def execute_attempt(db, folder, question, cfg, api_key, transport=http_transport):
    attempt = start_attempt(db, folder, question, cfg)
    headers = providers.headers_for(cfg, api_key, attempt["client_request_id"])
    started = time.monotonic()
    interrupted = False
    try:
        status, received_headers, body = transport(attempt["payload_json"].encode("utf-8"), headers, cfg["timeout_seconds"])
        # Normally OpenAI never echoes the API key. If it does, protect it and
        # explicitly mark that raw bytes were redacted, rather than claiming exactness.
        key_bytes = api_key.encode("utf-8")
        redacted_body = bool(key_bytes and key_bytes in body)
        if redacted_body:
            body = body.replace(key_bytes, b"[REDACTED]")
        receipt = {"http_status": status, "headers": redact({k.lower(): v for k, v in received_headers.items()}, api_key),
                   "body_base64": base64.b64encode(body).decode("ascii"), "body_sha256": sha(body),
                   "body_redacted_for_secret": redacted_body}
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        receipt = {"transport_error": {"error_type": type(exc).__name__, "error_code": "delivery_unknown",
                   "error_message": redact(str(exc) or "Interrupted", api_key),
                   "traceback": redact(traceback.format_exc(), api_key)}}
    receipt.update(provider=cfg["provider"], ended_at_utc=utc_now(), latency_ms=round((time.monotonic() - started) * 1000, 3))
    directory = Path(folder) / Path(attempt["request_path"]).parent
    atomic_json(directory / "receipt.json", receipt)
    result = finish_attempt(db, folder, attempt, receipt)
    if interrupted:
        raise KeyboardInterrupt
    return result, receipt


def retry_delay(cfg, receipt, local_attempt):
    delay = min(cfg["retry_max_seconds"], cfg["retry_base_seconds"] * 2 ** (local_attempt - 1))
    delay += random.uniform(0, min(1, delay * .1))
    header = receipt.get("headers", {}).get("retry-after")
    if header:
        try:
            after = float(header)
        except ValueError:
            try:
                after = (email.utils.parsedate_to_datetime(header) - dt.datetime.now(dt.timezone.utc)).total_seconds()
            except (TypeError, ValueError):
                after = 0
        delay = max(delay, after)
    return max(0, delay)


def responsive_wait(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(min(.5, max(0, end - time.monotonic())))


def run_pending(db, folder, cfg, api_key, limit=None, retry_errors=False, retry_unknown=False,
                transport=http_transport, wait=responsive_wait):
    recover(db, folder)
    states = ["PENDING"]
    if retry_errors:
        states += ["API_ERROR", "PROTOCOL_ERROR"]
    if retry_unknown:
        states += ["UNKNOWN"]
    rows = db.execute("SELECT q.*,r.repeat_index,r.status,r.latest_attempt_id FROM questions q JOIN repetitions r "
                      "ON q.question_id=r.question_id WHERE r.status IN (" + ",".join("?" for _ in states) +
                      ") ORDER BY q.ordinal,r.repeat_index", states).fetchall()
    if limit is not None:
        selected_ids = set(list(dict.fromkeys(r["question_id"] for r in rows))[:limit])
        rows = [r for r in rows if r["question_id"] in selected_ids]
    event(db, folder, "execution_started", selected_questions=len({r["question_id"] for r in rows}),
          selected_repetitions=len(rows), retry_errors=retry_errors,
          retry_unknown=retry_unknown, runtime=runtime_info())
    stop, processed, interrupted = False, 0, False
    processed_ids = set()
    try:
        for index, row in enumerate(rows):
            for local_attempt in range(1, cfg["max_attempts_per_execution"] + 1):
                result, receipt = execute_attempt(db, folder, row, cfg, api_key, transport)
                print(f"{row['question_id']}  repeat {row['repeat_index']}/{cfg['repetitions']}  {result['status']}  HTTP={result['http_status']}", flush=True)
                if result["stop_run"]:
                    stop = True
                    break
                if not result["retryable"] or local_attempt == cfg["max_attempts_per_execution"]:
                    break
                delay = retry_delay(cfg, receipt, local_attempt)
                event(db, folder, "retry_scheduled", question_id=row["question_id"], repeat_index=row["repeat_index"],
                      delay_seconds=delay, reason=result["error_code"] or result["http_status"])
                print(f"  Waiting {delay:.1f} seconds before retry.", flush=True)
                wait(delay)
            if not stop and result["status"] in {"API_ERROR", "PROTOCOL_ERROR", "UNKNOWN"}:
                event(db, folder, "question_error_skipped", question_id=row["question_id"],
                      repeat_index=row["repeat_index"], attempt_id=result["attempt_id"],
                      status=result["status"], error_code=result["error_code"],
                      retry_limit_reached=bool(result["retryable"]))
                print("  Error recorded. Continuing to the next pending item.", flush=True)
            processed += 1
            processed_ids.add(row["question_id"])
            last_for_question = index+1 == len(rows) or rows[index+1]["question_id"] != row["question_id"]
            if last_for_question and len(processed_ids) % cfg["checkpoint_every"] == 0:
                export_all(db, folder, tolerate_excel_lock=True)
            if stop:
                break
            if index+1 < len(rows):
                wait(cfg["min_interval_seconds"])
    except KeyboardInterrupt:
        interrupted = True
        event(db, folder, "user_interrupted")
        print("Interrupt requested; saving collected logs and Excel output.")
    except Exception as exc:
        event(db, folder, "local_execution_error", error_type=type(exc).__name__,
              error_message=redact(str(exc), api_key), traceback=redact(traceback.format_exc(), api_key))
        raise
    finally:
        event(db, folder, "execution_finished", processed_questions=len(processed_ids),
              processed_repetitions=processed, stopped_for_error=stop, interrupted=interrupted)
        export_all(db, folder, tolerate_excel_lock=True)
    return {"processed": processed, "processed_questions": len(processed_ids),
            "stopped_for_error": stop, "interrupted": interrupted}


def excel_text(value):
    """Only an Excel display copy is escaped/truncated; raw logs stay exact."""
    if not isinstance(value, str):
        return value, False
    safe = INVALID_XML.sub(lambda m: f"\\u{ord(m.group()):04x}", value)
    modified = safe != value
    if len(safe.encode("utf-16-le")) // 2 > 32767:
        suffix = "\n[Excel display truncated; see full_answer_path for the complete text]"
        budget = 32767 - len(suffix.encode("utf-16-le")) // 2
        safe = safe.encode("utf-16-le")[:budget * 2].decode("utf-16-le", errors="ignore") + suffix
        modified = True
    return safe, modified


def attempts_for_export(db):
    for row in db.execute("SELECT * FROM attempts ORDER BY started_at_utc,attempt_id"):
        value = dict(row)
        value["payload"] = json.loads(value.pop("payload_json"))
        value["receipt"] = json.loads(value.pop("receipt_json")) if value["receipt_json"] else None
        value["result"] = json.loads(value.pop("result_json")) if value["result_json"] else None
        yield value


def result_filename(cfg):
    """Include the requested model ID without allowing it to create a path.

    Filename-safe model IDs are preserved verbatim. Normalized/long IDs
    also carry a short digest so different server model IDs stay distinct.
    """
    model = cfg['model']
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', model).strip('._-')[:120]
    if safe != model:
        safe = (safe or 'model') + '-' + sha(model.encode('utf-8'))[:12]
    return f"{safe}_{cfg['condition']}_result.xlsx"


def export_all(db, folder, tolerate_excel_lock=False):
    folder = Path(folder)
    cfg, headers = get_meta(db, "config"), get_meta(db, "headers")
    run_id = get_meta(db, "run_id")
    attempts = list(attempts_for_export(db))
    amap = {a["attempt_id"]: a for a in attempts}
    counts = collections.Counter((a["question_id"], a["repeat_index"]) for a in attempts)
    identity = lambda a: {"run_id": run_id, "question_id": a["question_id"],
                          "repeat_index": a["repeat_index"], "attempt_id": a["attempt_id"]}
    jsonlines(folder / "requests.jsonl", ({**identity(a), **{k: a[k] for k in ["attempt_number",
              "started_at_utc", "status", "request_sha256", "client_request_id", "request_path", "payload"]}} for a in attempts))
    jsonlines(folder / "responses.jsonl", ({**identity(a), "receipt": a["receipt"], "result": a["result"]}
              for a in attempts if a["receipt"] is not None))
    jsonlines(folder / "errors.jsonl", ({**identity(a), "result": a["result"], "receipt": a["receipt"]}
              for a in attempts if a["result"] and a["result"]["status"] in {"API_ERROR", "UNKNOWN", "PROTOCOL_ERROR"}))
    jsonlines(folder / "events.jsonl", ({**json.loads(r["event_json"]), "seq": r["seq"]}
              for r in db.execute("SELECT * FROM events ORDER BY seq")))
    questions = db.execute("SELECT * FROM questions ORDER BY ordinal").fetchall()
    reps = db.execute("SELECT q.*,r.repeat_index,r.status,r.latest_attempt_id FROM questions q JOIN repetitions r "
                      "ON q.question_id=r.question_id ORDER BY q.ordinal,r.repeat_index").fetchall()
    all_results, detail_values, by_key = [], [], {}
    for q in reps:
        a = amap.get(q["latest_attempt_id"]) or {}
        r = a.get("result") or {}
        key = (q["question_id"], q["repeat_index"])
        usage = r.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        answer, modified = excel_text(r.get("answer", ""))
        meta = {**r, "attempt_count": counts[key], "latest_attempt_id": q["latest_attempt_id"],
                "requested_model": cfg["model"], "provider": cfg["provider"], "condition": cfg["condition"], "client_request_id": a.get("client_request_id"),
                "started_at_utc": a.get("started_at_utc"), "prompt_sha256": q["prompt_sha256"],
                "request_sha256": a.get("request_sha256"), "request_payload_path": a.get("request_path"),
                "cached_tokens": details.get("cached_tokens"), "excel_display_modified": "Y" if modified else "N",
                **{k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}, "status": q["status"]}
        by_key[key] = (answer, meta)
        all_results.append({"run_id": run_id, "question_id": q["question_id"], "repeat_index": q["repeat_index"],
                            "prompt_sha256": q["prompt_sha256"], "status": q["status"], "attempt_count": counts[key],
                            "latest_attempt_id": q["latest_attempt_id"], "result": a.get("result")})
        context = canonical_context(q)
        user_prompt, document = context["user_prompt"], context["document"]
        meta["user_prompt"] = user_prompt
        meta["document"] = document
        detail_values.append([q["question_id"], q["repeat_index"], q["prompt"], answer, user_prompt, document] + [meta.get(k) for k in EXTRA_COLUMNS])
    jsonlines(folder / "results.jsonl", all_results)
    usage_totals = collections.Counter()
    for a in attempts:
        usage = (a["result"] or {}).get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(usage.get(k), int):
                usage_totals[k] += usage[k]
    status_by_repeat = {str(rep): dict(collections.Counter(q["status"] for q in reps if q["repeat_index"] == rep))
                        for rep in range(1, cfg["repetitions"]+1)}
    summary = {"run_id": run_id, "updated_at_utc": utc_now(), "questions": len(questions),
               "repetitions_per_question": cfg["repetitions"], "planned_repetitions": len(reps),
               "attempts": len(attempts), "status_counts": dict(collections.Counter(q["status"] for q in reps)),
               "status_counts_by_repeat": status_by_repeat,
               "reported_token_totals_all_attempts": dict(usage_totals),
               "note": "Status counts count question/repetition pairs, not unique questions. Unknown requests may have been billed."}
    atomic_json(folder / "summary.json", summary)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = cfg["condition"] + "_result"
    pi = headers.index(cfg["prompt_column"])
    answer_headers = [f"{ANSWER_COLUMN}_r{i}" for i in range(1, cfg["repetitions"]+1)]
    compact_keys = ["status", "attempt_count", "response_id", "requested_model", "returned_model"]
    leading = [cfg["id_column"], cfg["prompt_column"]]
    out_headers = leading + answer_headers + [USER_PROMPT_COLUMN, DOCUMENT_COLUMN] + [h for h in headers if h not in {*leading, DOCUMENT_COLUMN}]
    out_headers += [f"{k}_r{i}" for i in range(1, cfg["repetitions"]+1) for k in compact_keys]
    ws.append(out_headers)
    for q in questions:
        original = json.loads(q["values_json"])
        results = [by_key[(q["question_id"], rep)] for rep in range(1, cfg["repetitions"]+1)]
        record = dict(zip(headers, original))
        record.update(zip(answer_headers, [item[0] for item in results]))
        record[USER_PROMPT_COLUMN] = results[0][1]["user_prompt"]
        record[DOCUMENT_COLUMN] = results[0][1]["document"]
        record.update({f"{key}_r{rep}": item[1].get(key) for rep, item in enumerate(results, 1) for key in compact_keys})
        values = [record.get(h) for h in out_headers]
        ws.append([excel_text(v)[0] for v in values])
    detail = wb.create_sheet("repetition_details")
    detail.append(["question_id", "repeat_index", cfg["prompt_column"], ANSWER_COLUMN, USER_PROMPT_COLUMN, DOCUMENT_COLUMN] + EXTRA_COLUMNS)
    for values in detail_values:
        detail.append([excel_text(v)[0] for v in values])
    for sheet, table_name in [(ws, "ExperimentResults"), (detail, "RepetitionDetails")]:
        for row in sheet:
            for cell in row:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.font = Font(name="Arial", size=10)
        for cell in sheet[1]:
            cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="223B57")
        sheet.row_dimensions[1].height = 36
        for row in range(2, sheet.max_row+1):
            sheet.row_dimensions[row].height = 90
        for col, cell in enumerate(sheet[1], 1):
            h = cell.value
            width = 78 if h == cfg["prompt_column"] else 92 if h.startswith(ANSWER_COLUMN) or h in {USER_PROMPT_COLUMN, DOCUMENT_COLUMN, PERSONA_COLUMN} else 24
            sheet.column_dimensions[get_column_letter(col)].width = width
        sheet.freeze_panes = "C2"
        table = Table(displayName=table_name, ref=sheet.dimensions)
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        sheet.add_table(table)
    info = wb.create_sheet("summary")
    for k, v in {**summary, "config": cfg, "provider_settings": providers.describe_settings(cfg), "original_input_sha256": get_meta(db, "input_sha256"),
                 "status_note": "OK indicates successful response receipt. Compliance is determined in the judge stage.",
                 "context_note": f"{cfg['condition']}: The request uses the configured system instruction and serialized tool history.",
                 "full_text_note": "user_prompt and document_text match the request payload. PENDING identifies prepared inputs awaiting generation.",
                 "reference_mapping_sha256": get_meta(db, "reference_mapping_sha256"),
                 "repeat_note": f"{cfg['repetitions']} repetitions per question. Each request is independent; retries are recorded separately.",
                 "error_policy": "Item-specific errors are recorded before continuing. Authentication, permission, quota, model-access and shared-parameter errors stop the run. UNKNOWN requests require an explicit retry."}.items():
        info.append([k, dumps(v) if isinstance(v, (dict, list)) else v])
    info.column_dimensions["A"].width = 38
    info.column_dimensions["B"].width = 105
    for row in info:
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    destination = folder.parent.parent / result_filename(cfg)
    tmp = destination.with_name(destination.stem + ".tmp.xlsx")
    try:
        wb.save(tmp)
        os.replace(tmp, destination)
    except PermissionError:
        if not tolerate_excel_lock:
            raise
        event(db, folder, "excel_export_failed", error_type="PermissionError", message="Close Excel, then run export.")
        print("Excel output could not be updated. Logs are saved. Close the workbook and export again.", file=sys.stderr)
    finally:
        wb.close()
        if tmp.exists():
            tmp.unlink()
    return summary


