"""Offline integration checks using mock provider responses and a mock tokenizer."""
from contextlib import closing, nullcontext, redirect_stderr, redirect_stdout
from functools import partial
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PipelineTest(unittest.TestCase):
    def test_all_models_then_four_judges(self):
        sys.path.insert(0, str(ROOT))
        import pipeline
        from suite.adapters import definitions

        expected_models = ["gpt-4.1-2025-04-14", "gpt-5.1-2025-11-13", "gpt-5.4-2026-03-05",
                           "gpt-5.6-luna", "claude-opus-4-6", "gemini-3.1-flash-lite"]
        conditions = ["c0", "c1", "c2", "c3_e", "c3_p", "c4"]
        judges = ["gpt56_luna_research", "harmbench_full", "strongreject", "deepseek_research"]
        events = []
        case = json.loads((ROOT / "input.json").read_text())
        expected_answers = [f"Reading club {model} {c}." for model in expected_models for c in conditions]
        fail_once = True

        def target_transport(cfg):
            model, condition = cfg["model"], cfg["condition"]
            def send(raw, headers, timeout):
                nonlocal fail_once
                body = json.loads(raw)
                events.append(("target", model, condition))
                if model == expected_models[1] and fail_once:
                    fail_once = False
                    return 401, {}, b'{"error":{"code":"invalid_api_key","message":"mock interruption"}}'
                expected_user = case["question"]
                if condition in {"c1", "c3_e", "c3_p", "c4"}:
                    expected_user = case["persona"] + "\n\n" + expected_user
                expected_docs = [] if condition in {"c0", "c1"} else [
                    {"id": "doc_01", "text": case["documents"]["c4" if condition == "c2" else condition]}]
                reply = f"Reading club {model} {condition}."
                if cfg["provider"] == "openai":
                    self.assertEqual("reasoning_effort" in body, model != expected_models[0])
                    self.assertEqual(body["messages"][-1]["role"], "tool")
                    self.assertEqual(body["messages"][1]["content"], expected_user)
                    self.assertEqual(json.loads(body["messages"][-1]["content"])["documents"], expected_docs)
                    result = {"id": "mock", "model": model, "choices": [
                        {"finish_reason": "stop", "message": {"content": reply}}]}
                elif cfg["provider"] == "anthropic":
                    self.assertEqual(body["messages"][-1]["content"][0]["type"], "tool_result")
                    self.assertEqual(body["messages"][0]["content"], expected_user)
                    self.assertEqual(json.loads(body["messages"][-1]["content"][0]["content"])["documents"], expected_docs)
                    result = {"id": "mock", "model": model, "stop_reason": "end_turn",
                              "content": [{"type": "text", "text": reply}]}
                else:
                    self.assertIn("functionResponse", body["contents"][-1]["parts"][0])
                    self.assertEqual(body["contents"][0]["parts"][0]["text"], expected_user)
                    self.assertEqual(body["contents"][-1]["parts"][0]["functionResponse"]["response"]["documents"], expected_docs)
                    result = {"responseId": "mock", "modelVersion": model, "candidates": [
                        {"finishReason": "STOP", "content": {"parts": [{"text": reply}]}}]}
                return 200, {}, json.dumps(result).encode()
            return send

        class CountingTokenizer:
            def encode(self, value, **kwargs):
                return list(range(len(value.split()) + 1))

        connections = {j: {"base_url": "https://test.invalid/" + j, "served_model_id": d["model"],
                           "key": "offline-test-key"} for j, d in definitions().items()}

        def judge_transport(url, body, key, timeout, attempt, **kwargs):
            judge = url.split("/")[3]
            serialized = json.dumps(body)
            matches = [answer for answer in expected_answers if answer in serialized]
            self.assertEqual(len(matches), 1)
            events.append(("judge", judge, matches[0]))
            if judge == "harmbench_full":
                choice = {"text": "Yes", "finish_reason": "stop"}
            else:
                answer = "1.b 0\n2.b 4\n3.b 3" if judge == "strongreject" else '{"label":"compliant"}'
                choice = {"message": {"content": answer}, "finish_reason": "stop"}
            return {"http_status": 200, "raw_body_text": json.dumps({"choices": [choice]})}

        def cli(*args):
            with patch.object(sys, "argv", ["pipeline.py", *map(str, args)]), redirect_stdout(io.StringIO()):
                pipeline.main()

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            jf = Path(d) / "judges.json"
            cfg = json.loads((ROOT / "judges/configs/judges.json").read_text())
            cfg["pause_seconds"] = 0
            jf.write_text(json.dumps(cfg))
            args = ("run", "--out", out, "--judge-config", jf)
            keys = {k: "target-offline-credential" for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")}
            execute = partial(pipeline.runner.execute, transport=judge_transport, resolved_for_test=connections,
                              tokenizers={"harmbench_full": CountingTokenizer()})
            fast_target = partial(pipeline.target.execute_all, wait=lambda seconds: None)
            with (patch.dict(os.environ, keys), patch.object(pipeline.target.providers, "transport_for", target_transport),
                  patch.object(pipeline.target, "execute_all", fast_target), patch.object(pipeline.runner, "execute", execute)):
                with self.assertRaisesRegex(ValueError, "unavailable judge"):
                    cli(*args, "--judges", "unknown_judge")
                self.assertFalse(out.exists())
                with self.assertRaisesRegex(RuntimeError, "stopped"):
                    cli(*args)
                self.assertEqual(len(events), 7)
                self.assertFalse((out / "judge").exists())
                cli("resume", "--out", out, "--retry-target-errors")
                expected_targets = [("target", model, cond) for model in expected_models for cond in conditions]
                expected_targets.insert(6, ("target", expected_models[1], "c0"))
                expected_events = expected_targets + [("judge", j, a) for j in judges for a in expected_answers]
                self.assertEqual(events, expected_events)
                cli("resume", "--out", out)
                self.assertEqual(events, expected_events)
                copied_case = out / "input.json"
                copied_case.write_text(copied_case.read_text() + " ")
                with self.assertRaisesRegex(ValueError, "compatible fresh run"):
                    cli("resume", "--out", out)
                self.assertEqual(events, expected_events)
            rows = [json.loads(line) for line in (out / "judge/inputs.normalized.jsonl").read_text().splitlines()]
            self.assertEqual(len({r["target_run"] for r in rows}), 6)
            self.assertEqual([r["answer"] for r in rows], expected_answers)
            self.assertEqual([r["target_model"] for r in rows], [m for m in expected_models for _ in conditions])
            results = [json.loads(line) for line in (out / "judge/logs/results.jsonl").read_text().splitlines()]
            self.assertEqual(len(results), 144)
            self.assertTrue(all(r["analysis_valid"] for r in results), results[:1])
            with closing(load_workbook(out / "judge/results.xlsx", read_only=True)) as book:
                hdr = next(book["Results"].values)
                self.assertFalse(any("unknown_judge" in str(col).lower() for col in hdr))

    def test_judge_profiles(self):
        sys.path.insert(0, str(ROOT / "judges"))
        from suite.adapters import Adapter, definitions
        from suite.common import verify_sources
        verify_sources()
        self.assertEqual(list(definitions()), ["gpt56_luna_research", "harmbench_full",
                                              "strongreject", "deepseek_research"])
        cases = [("gpt56_luna_research", '{"label":"compliant"}', "label", "compliant"),
                 ("deepseek_research", '{"label":"refusal"}', "label", "refusal"),
                 ("strongreject", "1.b 0\n2.b 4\n3.b 3", "score", 0.625)]
        for profile, raw, field, expected in cases:
            with self.subTest(profile=profile):
                adapter = Adapter(profile)
                route, payload, _ = adapter.build("Where does the club meet?", "Room 204.")
                self.assertEqual(route, "/chat/completions")
                self.assertIn("Room 204.", json.dumps(payload))
                mock = {"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]}
                result = adapter.interpret({"raw_body_text": json.dumps(mock), "http_status": 200})
                self.assertTrue(result["analysis_valid"])
                self.assertEqual(result[field], expected)

    def test_english_compliance_labels(self):
        import pipeline
        from suite.adapters import Adapter
        from suite.common import ORDER
        expected = {"refusal": 0, "clarification_request": 1, "undetermined": 2,
                    "partially_compliant": 3, "compliant": 4}
        self.assertEqual(ORDER, expected)
        for profile in ("gpt56_luna_research", "deepseek_research"):
            adapter = Adapter(profile)
            _, body, _ = adapter.build("Where does the club meet?", "Room 204.")
            schema = body["response_format"]["json_schema"]["schema"]
            self.assertEqual(set(schema["properties"]["label"]["enum"]), set(expected))
            for label in expected:
                with self.subTest(profile=profile, label=label):
                    raw = json.dumps({"choices": [{"finish_reason": "stop",
                        "message": {"content": json.dumps({"label": label})}}]})
                    result = adapter.interpret({"http_status": 200, "raw_body_text": raw})
                    self.assertTrue(result["analysis_valid"])
                    self.assertEqual(result["label"], label)

    def test_standalone_workbook_exports(self):
        import pipeline
        from suite.adapters import legacy
        from openpyxl import Workbook
        case = json.loads((ROOT / "input.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "input.xlsx"
            cfg = json.loads((ROOT / "judges/legacy/together/config.json").read_text())
            book = Workbook()
            sheet = book.active
            sheet.title = "input"
            sheet.append(["question_id", "question", *cfg["response_columns"].values()])
            sheet.append([case["question_id"], case["question"], *(["Example response."] * 6)])
            book.save(source)
            book.close()
            for component in ("together", "strongreject"):
                with self.subTest(component=component):
                    module = legacy(component)
                    spec = module.load_spec(ROOT / "judges/legacy" / component / "config.json", source)
                    audit = module.Audit(directory / component, spec)
                    try:
                        output = module.export_workbook(audit, spec)
                    finally:
                        audit.close()
                    self.assertTrue(Path(output).is_file())
                    with closing(load_workbook(output, read_only=True)) as result:
                        summaries = [name for name in result.sheetnames if name.endswith("summary")]
                        self.assertEqual(len(summaries), 1)
                        rows = list(result[summaries[0]].values)
                        self.assertTrue(any(row[0] == "rubric_version" for row in rows))
                        for ws in result:
                            self.assertFalse(any(0xAC00 <= ord(char) <= 0xD7A3 for char in ws.title))
                            for row in ws.values:
                                for value in row:
                                    if isinstance(value, str):
                                        self.assertFalse(any(0xAC00 <= ord(char) <= 0xD7A3 for char in value))

            # Exercise HarmBench's native loader and exporter without downloading a tokenizer.
            module = legacy("harmbench")
            cfg = module.load_config(ROOT / "judges/legacy/harmbench/config_full.json")
            _, rows = module.source_rows(source, cfg)
            run = directory / "harmbench"
            run.mkdir()
            import shutil
            shutil.copyfile(source, run / "input_snapshot.xlsx")
            tasks = [{"question_id": row["question_id"], "condition": condition,
                      "response_truncated": False, "question": row["question"],
                      "original_response": row["responses"][condition]}
                     for row in rows for condition in module.CONDITIONS]
            manifest = {"harmbench_git_commit": module.assets()["harmbench_git_commit"],
                        "input_sha256": module.sha_file(source), "fingerprint": "offline-export-test"}
            db = module.db_open(run)
            try:
                db.execute("CREATE TABLE results (question_id TEXT, condition TEXT, status TEXT, label TEXT, record TEXT)")
                for task in tasks:
                    record = {"status": "READY", "label": None}
                    db.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?)",
                               (task["question_id"], task["condition"], "READY", None, json.dumps(record)))
                db.commit()
                module.export_results(run, cfg, manifest, tasks, db)
            finally:
                db.close()
            self.assertTrue((run / "judge_results_with_harmbench.xlsx").is_file())

    def test_placeholder_credentials(self):
        import pipeline
        from suite.adapters import Adapter, legacy
        from suite.runner import connections

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for profile in ("deepseek_research", "strongreject"):
                adapter = Adapter(profile)
                for field in ("url", "key"):
                    with self.subTest(profile=profile, field=field):
                        cfg = {"connections": {profile: {
                            "url": "https://example.endpoints.huggingface.cloud/v1",
                            "key_env": "OFFLINE_JUDGE_KEY"}}, "timeout_seconds": 1}
                        key = "offline-test-key"
                        if field == "url":
                            cfg["connections"][profile]["url"] = "https://YOUR-ENDPOINT.endpoints.huggingface.cloud/v1"
                        else:
                            key = "YOUR-API-KEY"
                        transport = Mock(side_effect=AssertionError("A placeholder must not trigger an HTTP call"))
                        with patch.dict(os.environ, {"OFFLINE_JUDGE_KEY": key}, clear=True):
                            with self.assertRaisesRegex(ValueError, "placeholder"):
                                connections(cfg, {profile: adapter}, directory, transport=transport)
                        transport.assert_not_called()

            module = legacy("strongreject")
            endpoint = "https://example.endpoints.huggingface.cloud"
            config_path = ROOT / "judges/legacy/strongreject/config.json"
            with patch.object(module, "load_env"):
                for token, url in (("", endpoint), ("YOUR-HF-TOKEN", endpoint),
                                   ("offline-test-key", ""),
                                   ("offline-test-key", "https://YOUR-ENDPOINT.endpoints.huggingface.cloud")):
                    with self.subTest(token=token, url=url):
                        with patch.dict(os.environ, {"HF_TOKEN": token, "HF_ENDPOINT_URL": url}, clear=True):
                            with self.assertRaises(ValueError):
                                module.credentials(config_path)
                with patch.dict(os.environ, {"HF_TOKEN": "offline-test-key", "HF_ENDPOINT_URL": endpoint}, clear=True):
                    self.assertEqual(module.credentials(config_path), (endpoint, "offline-test-key"))

            module = legacy("together")
            for key in ("", "YOUR-TOGETHER-KEY", "offline-test-key"):
                with self.subTest(together_key=key):
                    with (patch.dict(os.environ, {"TOGETHER_API_KEY": key}, clear=True),
                          patch.object(module, "load_env"), patch.object(module, "load_spec", return_value={}),
                          patch.object(module, "describe", return_value={}),
                          patch.object(module, "run_lock", return_value=nullcontext()),
                          patch.object(module, "Audit", return_value=Mock()),
                          patch.object(module, "ensure_smoke", return_value={}) as smoke,
                          redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                        code = module.main(["smoke", "--run-dir", str(directory / "together")])
                    if key == "offline-test-key":
                        self.assertEqual(code, 0)
                        smoke.assert_called_once()
                    else:
                        self.assertEqual(code, 2)
                        smoke.assert_not_called()

    def test_conditions(self):
        sys.path.insert(0, str(ROOT / "judges"))
        from suite.common import condition
        self.assertEqual(condition("c3_e"), "C3-E")
        for alias in ("c3-e-new", "c3_e_new", "C3ENEW"):
            with self.assertRaises(ValueError):
                condition(alias)


    def test_single_input_and_public_files(self):
        import pipeline
        case = json.loads((ROOT / "input.json").read_text())
        self.assertIsInstance(case, dict)
        self.assertEqual(case["question_id"], "example_001")
        self.assertEqual(set(case["documents"]), {"c3_e", "c3_p", "c4"})
        self.assertEqual({p.name for p in ROOT.rglob("*.md")},
                         {"llm_evaluation_public_v1.4.2_2026-11-18_README.md"})
        self.assertFalse(list(ROOT.rglob("*.zip")))
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            invalid = folder / "two_cases.json"
            invalid.write_text(json.dumps([case, case]))
            with self.assertRaisesRegex(ValueError, "one case object"):
                pipeline.prepare_inputs(invalid, folder)


if __name__ == "__main__":
    unittest.main()
