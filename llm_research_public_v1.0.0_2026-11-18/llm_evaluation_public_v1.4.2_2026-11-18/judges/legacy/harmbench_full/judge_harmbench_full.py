"""Load the pinned HarmBench module and validate full-response mode."""
import hashlib
import importlib.util

ORIGINAL_SHA256 = "130ce7591b45015f59145ceaeef9d14ae005b0ab893d9eb8e8eb4dab1340acc7"


def load_runner(path):
    if not path.is_file():
        raise ValueError("The HarmBench implementation file is missing.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("The HarmBench implementation hash differs from this release.")
    spec = importlib.util.spec_from_file_location("harmbench_pinned", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_full_config(module, config):
    cfg = module.load_config(config)
    if cfg["response_mode"] != "full":
        raise ValueError("This adapter requires response_mode=full. Use config_full.json.")
    if module.CONTEXT_LIMIT != 2048:
        raise ValueError("Expected a total context limit of 2048 tokens.")
    return cfg


