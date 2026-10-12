"""Remote training on the shared Laya server; review, snapshots and the promotion gate stay in the controller."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .store import ControlError, Store, absolute, atomic_json, canonical, digest
from .training import passes_policy, rows_for

REMOTE_TASK = "compact-laya-focus-v2"
REMOTE_OWNER = "compact-laya"
REMOTE_TRAIN_KEYS = ("epochs", "micro_batch", "grad_accum", "freeze_encoder", "seed")
SPLITS = ("train", "calibration", "dev", "test")
JOB_ID = re.compile(r"[0-9a-f]{12}")
CLIENT_TIMEOUT_S = 120.0  # A 50 MB split upload does not fit the library's 10 s default.


@dataclass
class Remote:
    client: Any
    rows_to_jsonl: Callable[[list], str]


def poc_home() -> Path:
    return Path(os.environ.get("LAYA_POC_HOME") or Path.home() / "Documents" / "laya-poc").expanduser()


def load_client_module():
    path = poc_home() / "client" / "laya_poc_client.py"
    if not path.is_file():
        raise ControlError("laya-poc client not found; set LAYA_POC_HOME to a checkout containing client/")
    cached = sys.modules.get("laya_poc_client")
    if cached is not None and Path(getattr(cached, "__file__", "") or "").resolve() == path.resolve():
        return cached
    sys.modules.pop("laya_poc_client", None)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    importlib.invalidate_caches()
    return importlib.import_module("laya_poc_client")


def make_remote() -> Remote:
    module = load_client_module()
    return Remote(module.Client(timeout=CLIENT_TIMEOUT_S), module.rows_to_jsonl)


def call(function, *args, **kwargs):
    """Run one client call; library errors become controlled diagnostics without server text."""
    try:
        return function(*args, **kwargs)
    except ControlError:
        raise
    except Exception as exc:  # only the client library's own errors are translated
        names = {c.__name__ for c in type(exc).__mro__}
        if "ServerError" in names:
            status, code = getattr(exc, "status", "?"), getattr(exc, "code", "?")
            raise ControlError(f"remote server rejected the request: HTTP {status} {code}") from None
        if "LayaPocError" in names:
            raise ControlError("remote server unavailable") from None
        raise
