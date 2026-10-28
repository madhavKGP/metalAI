"""Shared helpers: paths, seeding, config, result logging."""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(name: str = "m1") -> dict:
    import yaml

    with open(CONFIGS / f"{name}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


EXPERIMENTS = ROOT / "experiments"
_EXP_ID: str | None = None


def start_experiment(script: str, args=None, cfg: dict | None = None) -> str:
    """Register a run. Experiment id comes from env var METALAI_EXP (e.g. 'E002'),
    set by OpenCode per task (see experiments/EXPERIMENTS.md); otherwise 'adhoc-<time>'.
    Saves the exact command, args, config and git commit to experiments/runs/<id>/<script>.json."""
    import sys

    global _EXP_ID
    _EXP_ID = os.environ.get("METALAI_EXP") or f"adhoc-{_dt.datetime.now():%Y%m%d-%H%M%S}"
    snap = {"exp_id": _EXP_ID, "script": script, "started": now(), "git_commit": git_commit(),
            "command": " ".join(sys.argv), "args": vars(args) if args is not None else None, "config": cfg}
    save_json(snap, EXPERIMENTS / "runs" / _EXP_ID / f"{script}.json")
    print(f"[experiment] {_EXP_ID} | {script} | commit {snap['git_commit']}")
    return _EXP_ID


def current_exp() -> str:
    """Experiment id of this run (after start_experiment), else METALAI_EXP, else 'adhoc'."""
    return _EXP_ID or os.environ.get("METALAI_EXP") or "adhoc"


def append_result(csv_path: Path, row: dict) -> None:
    """Append one result row (adds exp_id, timestamp, git commit). Handles new columns."""
    import pandas as pd

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"exp_id": _EXP_ID or "unregistered", "timestamp": now(), "git_commit": git_commit(), **row}
    new = pd.DataFrame([row])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(csv_path, index=False)


def require_cuda(allow_cpu: bool = False) -> str:
    """Training must run on GPU (laptop constraint, see decisions.md D-006)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        print("[WARN] CUDA not available - running on CPU because --allow-cpu was given.")
        return "cpu"
    raise SystemExit(
        "CUDA GPU not available. Training is GPU-only (decisions.md D-006). "
        "Check the PyTorch CUDA install (OPENCODE.md T1) or pass --allow-cpu for a tiny smoke test."
    )
