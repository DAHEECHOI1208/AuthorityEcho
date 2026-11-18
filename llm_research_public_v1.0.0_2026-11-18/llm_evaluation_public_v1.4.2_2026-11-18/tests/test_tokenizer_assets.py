"""Offline tests: external cache integrity, explicit download, and local loading."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "judges/legacy/harmbench"
spec = importlib.util.spec_from_file_location("cache_test_module", BASE / "tokenizer_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class TokenizerAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = Path(self.temp.name) / "cache"
        self.payloads = {name: ("offline fixture " + name).encode() for name in assets.FILES}
        self.rows = {name: {"url": "https://example.invalid/" + name, "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest()} for name, data in self.payloads.items()}
        self.manifest = patch.object(assets, "manifest", return_value=self.rows)
        self.manifest.start()

    def tearDown(self):
        self.manifest.stop()
        self.temp.cleanup()

    def opener(self, url, timeout):
        self.assertEqual(timeout, 60)
        return io.BytesIO(self.payloads[url.rsplit("/", 1)[1]])

    def test_download_verify_and_repeat_without_network(self):
        sender = Mock(side_effect=self.opener)
        result = assets.download(self.cache, sender)
        self.assertEqual(sender.call_count, 4)
        self.assertEqual(set(result["verified_files"]), set(assets.FILES))
        self.assertEqual(assets.verify_cache(self.cache), self.cache)
        again = assets.download(self.cache, sender)
        self.assertEqual(sender.call_count, 4)
        self.assertEqual(again["downloaded_files"], [])
        self.assertFalse(list(self.cache.glob("*.part")))

    def test_missing_cache_has_no_implicit_network(self):
        with patch.object(assets.urllib.request, "urlopen") as sender:
            with self.assertRaisesRegex(ValueError, "Missing tokenizer"):
                assets.verify_cache(self.cache)
            sender.assert_not_called()

    def test_tampered_cache_is_rejected_before_download(self):
        assets.download(self.cache, self.opener)
        target = self.cache / "tokenizer.model"
        target.write_bytes(b"changed")
        with patch.object(assets.urllib.request, "urlopen") as sender:
            with self.assertRaisesRegex(ValueError, "mismatch"):
                assets.download(self.cache)
            sender.assert_not_called()
        self.assertEqual(target.read_bytes(), b"changed")

    def test_wrong_bytes_are_not_committed(self):
        row = self.rows[assets.FILES[0]]
        with self.assertRaisesRegex(ValueError, "mismatch"):
            assets.download(self.cache, lambda *a, **k: io.BytesIO(b"x" * row["bytes"]))
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_oversized_download_is_rejected(self):
        row = self.rows[assets.FILES[0]]
        with self.assertRaisesRegex(ValueError, "exceeds pinned size"):
            assets.download(self.cache, lambda *a, **k: io.BytesIO(b"x" * (row["bytes"] + 1)))
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_network_failure_leaves_prior_verified_members_reusable(self):
        count = 0
        def sender(url, timeout):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("simulated interruption")
            return self.opener(url, timeout)
        with self.assertRaisesRegex(OSError, "simulated interruption"):
            assets.download(self.cache, sender)
        self.assertEqual(len(list(self.cache.iterdir())), 1)
        sender2 = Mock(side_effect=self.opener)
        assets.download(self.cache, sender2)
        self.assertEqual(sender2.call_count, 3)

    def test_public_source_cannot_be_used_as_cache(self):
        with self.assertRaisesRegex(ValueError, "outside the public source"):
            assets.cache_directory(ROOT / "downloaded_tokenizer")

    def test_custom_environment_cache(self):
        with patch.dict(os.environ, {assets.CACHE_ENV: str(self.cache)}):
            self.assertEqual(assets.cache_directory(), self.cache)

    def test_release_manifest_has_unchanged_resource_identity(self):
        self.manifest.stop()
        rows = assets.manifest()
        self.assertEqual(rows["tokenizer.model"]["sha256"],
                         "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347")
        self.assertEqual(rows["tokenizer.model"]["bytes"], 499723)
        self.assertTrue(all(assets.REVISION in r["url"] for r in rows.values()))
        self.manifest.start()

    def test_runtime_verifies_cache_before_importing_tokenizer(self):
        module_spec = importlib.util.spec_from_file_location("hb_runtime_cache_test", BASE / "judge_harmbench_hf.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        with patch.dict(os.environ, {assets.CACHE_ENV: str(self.cache)}):
            with self.assertRaisesRegex(ValueError, "Missing tokenizer"):
                module.tokenizer()

    def test_runtime_loader_is_local_only_after_verification(self):
        module_spec = importlib.util.spec_from_file_location("hb_local_loader_test", BASE / "judge_harmbench_hf.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        cache_module = types.SimpleNamespace(verify_cache=Mock(return_value=self.cache))
        token = types.SimpleNamespace(truncation_side="left")
        factory = Mock(return_value=token)
        transformer = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=factory))
        fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
        with (patch.object(module.importlib.util, "spec_from_file_location", return_value=fake_spec),
              patch.object(module.importlib.util, "module_from_spec", return_value=cache_module),
              patch.dict(sys.modules, {"transformers": transformer})):
            self.assertIs(module.tokenizer(), token)
        cache_module.verify_cache.assert_called_once_with()
        factory.assert_called_once_with(self.cache, local_files_only=True, trust_remote_code=False)
        self.assertEqual(token.truncation_side, "right")


if __name__ == "__main__":
    unittest.main()
