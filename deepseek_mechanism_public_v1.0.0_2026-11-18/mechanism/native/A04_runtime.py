import hashlib
import json

class TaskFatal(RuntimeError):
    """State invalid for all remaining items in this task."""

def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'), default=str)

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()
