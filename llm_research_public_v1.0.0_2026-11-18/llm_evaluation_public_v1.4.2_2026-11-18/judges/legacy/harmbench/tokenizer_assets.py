"""Explicit, hash-checked setup of the unchanged external HarmBench tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

BASE = Path(__file__).resolve().parent
MODEL = "cais/HarmBench-Llama-2-13b-cls"
REVISION = "bda705349d1144fa618770bea64d99ce54e3835b"
FILES = ("config.json", "special_tokens_map.json", "tokenizer.model", "tokenizer_config.json")
CACHE_ENV = "LLM_HARMBENCH_TOKENIZER_CACHE"


def manifest():
    data = json.loads((BASE / "assets/provenance.json").read_text(encoding="utf-8"))
    if data["model_revision"] != REVISION or set(data["external_tokenizer_sha256"]) != set(FILES):
        raise ValueError("External tokenizer identity differs from this release.")
    result = {}
    for name in FILES:
        row = data["files"][name]
        expected_url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        if (row["url"] != expected_url or row["sha256"] != data["external_tokenizer_sha256"][name]
                or len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"])
                or not isinstance(row["bytes"], int) or not 0 < row["bytes"] <= 2000000):
            raise ValueError(f"Invalid pinned tokenizer manifest: {name}")
        result[name] = row
    return result


def cache_directory(path=None):
    if path is None:
        path = os.environ.get(CACHE_ENV)
    if not path:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        path = cache_root / "llm-research-public" / "harmbench" / REVISION
    path = Path(path).expanduser().resolve()
    package = BASE.parents[2]
    protected = package.parent if (package.parent / "release_manifest.json").exists() else package
    if path == protected or protected in path.parents:
        raise ValueError("Tokenizer cache must be outside the public source tree.")
    return path


def validate_file(path, row):
    if not path.is_file():
        raise ValueError(f"Missing tokenizer file: {path}. Run setup_harmbench_tokenizer.py download first.")
    if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"Tokenizer size/hash mismatch: {path}. Preserve it and choose a clean external cache.")


def verify_cache(path=None):
    path = cache_directory(path)
    for name, row in manifest().items():
        validate_file(path / name, row)
    return path


def download(path=None, opener=None):
    """Only this explicitly invoked function accesses the network; no credentials."""
    path = cache_directory(path)
    rows = manifest()
    path.mkdir(parents=True, exist_ok=True)
    # Reject changed cached files before downloading any missing member.
    for name, row in rows.items():
        if (path / name).exists() or (path / name).is_symlink():
            validate_file(path / name, row)
    opener = opener or urllib.request.urlopen
    fetched = []
    for name, row in rows.items():
        dest = path / name
        if dest.is_file():
            continue
        fd, temporary = tempfile.mkstemp(prefix=".tokenizer-", suffix=".part", dir=path)
        temporary = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as out:
                with opener(row["url"], timeout=60) as response:
                    if getattr(response, "status", 200) != 200:
                        raise ValueError("Tokenizer download returned a non-200 response.")
                    total = 0
                    while True:
                        chunk = response.read(min(65536, row["bytes"] + 1 - total))
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > row["bytes"]:
                            raise ValueError(f"Tokenizer download exceeds pinned size: {name}")
                        out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            validate_file(temporary, row)
            # Never silently replace a concurrently supplied different file.
            if dest.exists() or dest.is_symlink():
                validate_file(dest, row)
            else:
                try:
                    os.link(temporary, dest)
                except FileExistsError:
                    validate_file(dest, row)
            fetched.append(name)
        finally:
            temporary.unlink(missing_ok=True)
    verify_cache(path)
    return {"model": MODEL, "revision": REVISION, "cache": str(path),
            "downloaded_files": fetched, "verified_files": list(rows),
            "target_or_judge_api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "check"))
    parser.add_argument("--cache-dir", help=f"External cache directory; otherwise {CACHE_ENV} or the user cache.")
    args = parser.parse_args()
    if args.command == "download":
        result = download(args.cache_dir)
    else:
        result = {"cache": str(verify_cache(args.cache_dir)), "revision": REVISION,
                  "verified_files": list(FILES), "network_requests": 0}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
