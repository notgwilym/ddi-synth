"""Rebuild tutorial/results.json from the committed run records.

Run from the repo root:

    python tutorial/build_results.py

Every figure the notebook quotes is selected here by an explicit rule over runs/*.json,
and the ids of the runs behind it are stored beside it. If a number in the notebook is
ever questioned, the answer is the selector below and the run files it names, not
anyone's recollection of what a run produced.

The selectors match on the `dataset` field of the run config and on the free-text notes,
because that is how the runs were labelled at the time. Where a figure rests on a single
seed or on a dataset other than the one its name suggests, a `caveat` says so.
"""
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results.json"


def load_runs():
    runs = []
    for p in sorted((REPO / "runs").glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and "metrics" in r:
            runs.append(r)
    return runs


def summarise(rs):
    rs = sorted(rs, key=lambda r: r["timestamp"])
    if not rs:
        return {"f1": None, "n_seeds": 0, "run_ids": []}
    f = [r["metrics"]["micro_f1_pos"] for r in rs]
    return {"f1": round(st.mean(f), 3),
            "sd": round(st.stdev(f), 3) if len(f) > 1 else None,
            "p": round(st.mean(r["metrics"]["micro_p_pos"] for r in rs), 3),
            "r": round(st.mean(r["metrics"]["micro_r_pos"] for r in rs), 3),
            "n_seeds": len(rs),
            "run_ids": [r["run_id"] for r in rs],
            "first_run": rs[0]["timestamp"][:10]}


def main():
    runs = load_runs()
    D = lambda r: r["config"].get("dataset")
    N = lambda r: r.get("notes") or ""
    pick = lambda pred: summarise([r for r in runs if pred(r)])

    res = {
        "_note": "Every number is recomputed from committed runs/*.json by "
                 "tutorial/build_results.py. See run_ids.",
        "trajectory": {
            "v13": {**pick(lambda r: r["run_id"] == "20260727-122000-6d8adf"),
                    "caveat": "single seed; trained on composed dataset "
                              "20260727-121745-bafaa8 (padded to 85% NONE), "
                              "not the raw v13 generation"},
            "v14": pick(lambda r: D(r) == "v14-full" and "verifier pruning" in N(r)),
            "v15": pick(lambda r: D(r) == "v15" and N(r) == "v15"),
            "v17": pick(lambda r: D(r) == "v17" and N(r).startswith("v17")),
            "v18": pick(lambda r: D(r) == "v18" and N(r) == "v18"),
        },
        "three_arm": {
            "human": pick(lambda r: D(r) == "human-matched"),
            "llm_labels": pick(lambda r: D(r) == "label-high-c8"),
            "synthetic": pick(lambda r: D(r) == "v18-low-matched"),
            "synthetic_high_effort": pick(lambda r: D(r) == "v18-high-matched"),
        },
        "exchange_rate": {
            b: {"human": pick(lambda r, b=b: D(r) == f"xr-human-{b}"),
                "llm_labels": pick(lambda r, b=b: D(r) == f"xr-llm-labels-{b}")}
            for b in [100, 250, 500, 1000, 2248]},
        "human_baselines": {
            "70/15/15 winning config": pick(lambda r: D(r) == "human" and "winning" in N(r)),
            "markers aligned": pick(lambda r: D(r) == "human"
                                    and N(r) == "masking human/markers/aligned"),
            "8 monster sentences removed": pick(lambda r: D(r) == "human-filtered"),
        },
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT} from {len(runs)} run records")


if __name__ == "__main__":
    main()
