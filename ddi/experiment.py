import json, os, subprocess, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

def _git(*args, default=""):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default

def log_run(config, metrics, notes="", train_id=None, eval_id=None):
    RUNS.mkdir(exist_ok=True)
    diff = _git("diff", "HEAD")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_id": train_id,   # dataset_id of training data (None = reproducible human from code+seed)
        "eval_id": eval_id,     # dataset_id of the eval set the metrics were computed on
        "config": config,
        "metrics": metrics,
        "notes": notes,
        "git_sha": _git("rev-parse", "HEAD", default="NO_GIT"),
        "git_dirty": bool(diff),
    }
    if diff:
        (RUNS / "diffs").mkdir(exist_ok=True)
        (RUNS / "diffs" / f"{run_id}.patch").write_text(diff)

    path = RUNS / f"{run_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)          # atomic
    print(f"logged {run_id}{'  [DIRTY]' if record['git_dirty'] else ''}")
    return run_id

def load_runs():
    import pandas as pd
    rows = []
    for p in sorted(RUNS.glob("*.json")):
        r = json.loads(p.read_text())
        cfg = dict(r["config"])
        flat = {
            "run_id": r["run_id"],
            "train_id": r.get("train_id") or cfg.pop("train_id", None),
            "eval_id": r.get("eval_id") or cfg.pop("eval_id", None),
            "git_sha": r["git_sha"][:8],
            "git_dirty": r["git_dirty"],
        }
        flat.update({f"cfg.{k}": v for k, v in cfg.items()})
        flat.update({f"m.{k}": v for k, v in r["metrics"].items()})
        rows.append(flat)
    return pd.DataFrame(rows)