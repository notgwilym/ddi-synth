"""Collect every logged run into seed-combined experiments, ordered by date.

WHAT AN EXPERIMENT IS HERE
--------------------------
log_run writes one file per training run, and a run is one seed. What you actually want
to read is the experiment: the same configuration at several seeds. So runs are grouped
on every config field except `seed`, plus train_id, eval_id and notes.

Notes are part of the key on purpose. Two runs with an identical config and different
notes were two different intentions, and collapsing them would lose exactly the thing
this is being built to recover. The cost is that a typo in the notes splits a group, so
`--loose` drops notes from the key and is there for when that happens.

WHAT IS NOT DONE
----------------
No filtering, no dropping, no judgement about which runs "count". Superseded runs, failed
arms and one-seed probes are all kept and marked. An experiment log that quietly omits
the runs that went nowhere is not a log.

Seed variance is reported as sd and n. At n=1 sd is None rather than 0, because those are
different statements and v17's seed sd of 0.035 means a single seed carries no weight.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# fields that identify a configuration rather than a repetition of it
SEED_FIELDS = {"seed"}

HEADLINE = ["micro_f1_pos", "micro_p_pos", "micro_r_pos"]
EXTRA = ["macro_f1_pos", "train_size", "val_size", "train_time",
         "micro_f1_pos_DrugBank", "micro_f1_pos_MedLine"]
PER_LABEL = ["ADVISE", "EFFECT", "INT", "MECHANISM"]


def load_runs(runs_dir, include_archive=True):
    """Every run record on disk. Returns (records, skipped).

    Four files under runs/ are hand-written summary lists rather than run records. They
    are skipped and reported rather than silently ignored, because a file that looks like
    a run and is not is worth knowing about.
    """
    runs_dir = Path(runs_dir)
    paths = sorted(runs_dir.glob("*.json"))
    if include_archive:
        paths += sorted(runs_dir.glob("archive/*/*.json"))

    out, skipped = [], []
    for p in paths:
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            skipped.append((str(p), f"unparseable: {e}"))
            continue
        if not isinstance(r, dict) or "metrics" not in r or "config" not in r:
            skipped.append((str(p), "not a run record"))
            continue
        r["_path"] = str(p.relative_to(runs_dir.parent))
        r["_archived"] = "archive/" in r["_path"]
        out.append(r)
    return out, skipped


def load_manifests(manifest_dir):
    man = {}
    for p in sorted(Path(manifest_dir).glob("*.json")):
        try:
            m = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "dataset_id" in m:
            man[m["dataset_id"]] = m
    return man


def group_key(rec, use_notes=True):
    cfg = {k: v for k, v in rec["config"].items() if k not in SEED_FIELDS}
    parts = [("cfg", tuple(sorted((k, json.dumps(v, sort_keys=True))
                                  for k, v in cfg.items()))),
             ("train_id", rec.get("train_id")),
             ("eval_id", rec.get("eval_id")),
             ("archived", rec["_archived"])]
    if use_notes:
        parts.append(("notes", (rec.get("notes") or "").strip()))
    return json.dumps(parts, sort_keys=True)


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "sd": None, "n": 0, "min": None, "max": None}
    return {"mean": statistics.fmean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "n": len(vals),
            "min": min(vals), "max": max(vals)}


def describe(rec, manifests):
    """Short human label for the experiment, and where its training data came from."""
    cfg = rec["config"]
    name = cfg.get("dataset") or "?"
    bits = []
    if cfg.get("budget_sentences") is not None:
        bits.append(f"budget={cfg['budget_sentences']}")
    if cfg.get("synth_ratio") is not None:
        bits.append(f"ratio={cfg['synth_ratio']}")
    if cfg.get("prune_fraction") is not None:
        bits.append(f"prune={cfg['prune_fraction']}")
    if cfg.get("match_balance"):
        bits.append("balanced")
    if cfg.get("render_mode") and cfg["render_mode"] != "markers":
        bits.append(cfg["render_mode"])

    # The 22 July human sweep is ten experiments all called "human". Surface any
    # hyperparameter that departs from the settled config so they are distinguishable.
    for k, default in (("lr", 2e-5), ("epochs", 3), ("batch_size", 32),
                       ("max_length", 256), ("neg_ratio", None)):
        if k in cfg and cfg[k] != default:
            bits.append(f"{k}={cfg[k]}")

    did = rec.get("train_id") or cfg.get("synth_id")
    man = manifests.get(did) if did else None
    prov = None
    if man:
        gen = man.get("generator") or {}
        prov = {"dataset_id": did,
                "provenance": man.get("provenance"),
                "version": gen.get("version"),
                "gen_id": gen.get("gen_id"),
                "model": gen.get("model"),
                "reasoning_effort": gen.get("reasoning_effort"),
                "prompt_sha": gen.get("prompt_sha"),
                "n_sentences": (man.get("size") or {}).get("n_sentences"),
                "n_instances": (man.get("size") or {}).get("n_instances"),
                "label_distribution": man.get("label_distribution"),
                "created_at": man.get("created_at"),
                "notes": man.get("notes")}
    elif did:
        prov = {"dataset_id": did, "provenance": "manifest not found"}
    return name, bits, prov


def build(runs_dir="runs", manifest_dir="datasets/manifests",
          include_archive=True, use_notes=True):
    records, skipped = load_runs(runs_dir, include_archive)
    manifests = load_manifests(manifest_dir)

    groups = defaultdict(list)
    for r in records:
        groups[group_key(r, use_notes)].append(r)

    experiments = []
    for key, recs in groups.items():
        recs.sort(key=lambda r: r["timestamp"])
        first, last = recs[0], recs[-1]
        name, bits, prov = describe(first, manifests)

        metrics = {}
        for m in HEADLINE + EXTRA:
            a = agg([r["metrics"].get(m) for r in recs])
            if a["n"]:
                metrics[m] = a
        per_label = {}
        for lab in PER_LABEL:
            a = agg([r["metrics"].get(f"f1_{lab}") for r in recs])
            sup = agg([r["metrics"].get(f"support_{lab}") for r in recs])
            if a["n"]:
                per_label[lab] = {"f1": a["mean"], "support": sup["mean"]}

        seeds = sorted({r["config"].get("seed") for r in recs
                        if r["config"].get("seed") is not None})
        shas = sorted({r["git_sha"][:8] for r in recs})

        experiments.append({
            "name": name,
            "qualifiers": bits,
            "first_run": first["timestamp"],
            "last_run": last["timestamp"],
            "date": first["timestamp"][:10],
            "n_seeds": len(recs),
            "seeds": seeds,
            "notes": sorted({(r.get("notes") or "").strip() for r in recs}),
            "metrics": metrics,
            "per_label": per_label,
            "config": {k: v for k, v in first["config"].items() if k not in SEED_FIELDS},
            "train_id": first.get("train_id"),
            "eval_id": first.get("eval_id"),
            "provenance": prov,
            "git_shas": shas,
            "git_dirty": any(r["git_dirty"] for r in recs),
            "archived": first["_archived"],
            "run_ids": [r["run_id"] for r in recs],
            "paths": [r["_path"] for r in recs],
        })

    experiments.sort(key=lambda e: e["first_run"])
    return {"experiments": experiments,
            "manifests": manifests,
            "skipped": skipped,
            "n_runs": len(records)}


def summary(data):
    exps = data["experiments"]
    print(f"{data['n_runs']} run records -> {len(exps)} experiments")
    if data["skipped"]:
        print(f"{len(data['skipped'])} files skipped:")
        for p, why in data["skipped"]:
            print(f"    {p}: {why}")
    one_seed = sum(1 for e in exps if e["n_seeds"] == 1)
    dirty = sum(1 for e in exps if e["git_dirty"])
    print(f"{one_seed} experiments at a single seed, {dirty} logged against a dirty tree")
    by_day = defaultdict(int)
    for e in exps:
        by_day[e["date"]] += 1
    print(f"{len(by_day)} distinct days, "
          f"{min(by_day)} to {max(by_day)}")

    print(f"\n{'date':<11} {'n':>3} {'F1':>6} {'sd':>6} {'P':>6} {'R':>6}  experiment")
    for e in exps:
        f = e["metrics"].get("micro_f1_pos", {})
        p = e["metrics"].get("micro_p_pos", {})
        r = e["metrics"].get("micro_r_pos", {})
        sd = f"{f['sd']:.3f}" if f.get("sd") is not None else "   -  "
        label = e["name"] + (f" [{' '.join(e['qualifiers'])}]" if e["qualifiers"] else "")
        print(f"{e['date']:<11} {e['n_seeds']:>3} {f.get('mean', 0):>6.3f} {sd:>6} "
              f"{p.get('mean', 0):>6.3f} {r.get('mean', 0):>6.3f}  {label}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--manifests", default="datasets/manifests")
    ap.add_argument("--out", default="reports/experiments.json")
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--loose", action="store_true",
                    help="group without notes, for when a typo split a group")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    data = build(a.runs, a.manifests,
                 include_archive=not a.no_archive, use_notes=not a.loose)
    if not a.quiet:
        summary(data)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
