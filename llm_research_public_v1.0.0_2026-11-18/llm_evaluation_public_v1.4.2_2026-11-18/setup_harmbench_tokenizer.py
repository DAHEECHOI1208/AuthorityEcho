#!/usr/bin/env python3
"""Download/check pinned tokenizer assets outside the public package."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "judges/legacy/harmbench/tokenizer_assets.py"), run_name="__main__")
