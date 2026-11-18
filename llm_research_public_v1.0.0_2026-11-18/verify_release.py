#!/usr/bin/env python3
"""Offline integrity and decoded-string publication checks for this source bundle."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def is_hangul(text):
    ranges = ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F),
              (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF))
    return any(any(a <= ord(c) <= b for a, b in ranges) for c in text)


def json_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, val in value.items():
            yield str(key)
            yield from json_strings(val)
    elif isinstance(value, list):
        for val in value:
            yield from json_strings(val)


def check():
    failures = []
    declared = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(None, 1)
        relative = relative.lstrip("*")
        path = ROOT / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
            raise ValueError("Unsafe checksum entry")
        if relative in declared:
            raise ValueError("Duplicate checksum entry")
        declared[relative] = expected
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(relative + ": missing or changed")
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()}
    failures.extend(p + ": undeclared file" for p in sorted(actual - set(declared) - {"SHA256SUMS"}))
    credential = re.compile(r"(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----)")
    py_count = json_count = 0
    for relative in sorted(actual):
        path = ROOT / relative
        if path.is_symlink():
            failures.append(relative + ": symlink")
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            failures.append(relative + ": binary asset is not part of this release")
            continue
        strings = [text, relative]
        if path.suffix == ".py":
            tree = ast.parse(text)
            strings.extend(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
            py_count += 1
        elif path.suffix == ".json":
            strings.extend(json_strings(json.loads(text)))
            json_count += 1
        if any(is_hangul(s) for s in strings):
            failures.append(relative + ": Hangul in content or decoded strings")
        if any(credential.search(s) for s in strings):
            failures.append(relative + ": credential literal pattern")
    release = json.loads((ROOT / "release_manifest.json").read_text())
    mechanism = ROOT / release["packages"]["mechanism"]
    evaluation = ROOT / release["packages"]["evaluation"]
    left, right = (mechanism / "input.json").read_bytes(), (evaluation / "input.json").read_bytes()
    if left != right or hashlib.sha256(left).hexdigest() != release["input_sha256"]:
        failures.append("Bundled inputs differ or changed")
    case = json.loads(left)
    wrapper = json.loads((mechanism / "examples/benign_corpus.json").read_text())
    if wrapper["cases"][0]["input"] != case or wrapper["cases"][0]["metadata"]["variant"] != "benign":
        failures.append("Benign corpus wrapper differs")
    result = {"passed": not failures, "files": len(actual), "python_files_parsed": py_count,
              "json_files_parsed": json_count, "input_type": "benign",
              "network_requests": 0, "failures": failures,
              "scope": "Static release checks; not proof against all indirect identity inference or a runtime validation."}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    check()
