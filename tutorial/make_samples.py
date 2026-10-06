"""Extract small, committable samples of each generator version's real output.

Run once on the pod, from the repo root:

    python tutorial/make_samples.py

The tutorial never regenerates a dataset to make a point. It reads these samples, which
are drawn from the datasets the project actually trained on, so every structural claim
the notebook makes is checkable against the data that produced the logged F1.

Each sample file holds one line per sentence rather than one per pair, because the
tutorial's arguments are about sentences: how many drugs they carry, which construction
built them, and how the pair labels inside them are distributed. A pair-per-line file
would force every reader to regroup before seeing anything.

The generation spec is joined on where the raw file survives. Instance sent_ids are
`synth:{gen_id}:{spec_index}:s{n}` and raw records carry `spec_index`, so stripping the
`:s{n}` suffix gives the join key. Where the raw file is gone the sentence is still
emitted, with spec set to null, and samples/manifest.json records which versions lost
their specs. Nothing is silently dropped.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ddi.manifest import load_dataset
from ddi.synth import RAW
from ddi.verify_binary import group_by_sentence, _sentence_spans, _strip, _marked_spans, load_verdicts

# The canonical dataset per version: the one each version's headline F1 was measured on.
VERSIONS = {
    "v13": {"dataset_id": "20260727-121735-098a21", "gen_id": "v13"},
    "v14": {"dataset_id": "20260807-123340-ff79db", "gen_id": "v14-full-2"},
    "v15": {"dataset_id": "20260816-005908-3d6539", "gen_id": "v15-full"},
    "v17": {"dataset_id": "20260818-100023-0da98b", "gen_id": "v17-full"},
    "v18": {"dataset_id": "20260825-095703-67c1fa", "gen_id": "v18-full"},
}

# Spec fields worth keeping. Everything else is either derivable or prompt plumbing
# (register_line, axes) that would bloat the committed files without helping a reader.
SPEC_KEEP = ["frame", "kind", "assertions", "modifiers", "variants", "positives",
             "roles", "says", "shape", "register", "scene"]

_SUFFIX = re.compile(r":s\d+$")


def find_raw(version, cfg, instances):
    """The generation id whose raw file holds this dataset's specifications.

    Sentence ids encode it as synth:{gen_id}:{spec_index}:s{n}, which is more reliable
    than the table above: v13's dataset was built from a raw file not named v13.jsonl.
    Returns None, after listing likely files, if nothing matches."""
    from_ids = Counter(r["sent_id"].split(":")[1] for r in instances
                       if r["sent_id"].startswith("synth:"))
    for g in dict.fromkeys([g for g, _ in from_ids.most_common()] + [cfg["gen_id"]]):
        if (RAW / f"{g}.jsonl").exists():
            return g
    near = sorted(p.name for p in RAW.glob(f"*{version.lstrip('v')}*.jsonl"))
    print(f"{version}: no raw file for gen ids {list(from_ids)[:5]}; files that look close: {near[:10]}")
    return None


def read_specs(gen_id, render=None):
    """spec_index -> trimmed spec. Empty dict if the raw file did not survive.

    With a renderer, each trimmed spec also keeps `prompt`: the user message the model
    was sent, rebuilt from the full specification (rendering is deterministic)."""
    path = RAW / f"{gen_id}.jsonl"
    if not path.exists():
        return {}, False
    out = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("error") or not rec.get("sample"):
            continue
        spec = rec.get("spec") or {}
        idx = rec.get("spec_index", spec.get("spec_index"))
        if idx is None:
            continue
        trimmed = {k: spec[k] for k in SPEC_KEEP if k in spec}
        if render is not None:
            trimmed["prompt"] = render(spec)
        trimmed["entities"] = [e.get("surface") for e in spec.get("entities") or []]
        out[idx] = trimmed
    return out, True


def sentence_record(version, cfg, sent_id, rows, specs):
    spans = _sentence_spans(rows)
    base = _SUFFIX.sub("", sent_id)
    idx = base.rsplit(":", 1)[-1]
    spec = specs.get(int(idx)) if idx.isdigit() else None
    return {
        "version": version,
        "dataset_id": cfg["dataset_id"],
        "gen_id": cfg["gen_id"],
        "sent_id": sent_id,
        "sentence": _strip(rows[0]["text"]),
        "n_entities": len(spans),
        "pairs": [{"text": r["text"], "label": r["label"]} for r in rows],
        "spec": spec,
    }


def prompt_reproducible(version, manifest):
    """Whether the prompt that made this dataset can be re-rendered from current code.

    prompt.py was edited in place from v14 through v17, so only the last version written
    to each module is reproducible. The manifest stores the prompt_sha at generation time,
    which is enough to detect this even though it is not enough to recover the old text.
    """
    recorded = (manifest.get("generator") or {}).get("prompt_sha")
    if not recorded:
        return None
    try:
        if version == "v18":
            from ddi.prompt_v18 import fingerprint
        else:
            from ddi.prompt import fingerprint
        return fingerprint() == recorded
    except Exception:
        return None


def main(out_dir, n_sentences, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for version, cfg in VERSIONS.items():
        instances, manifest = load_dataset(cfg["dataset_id"])
        gen_id = find_raw(version, cfg, instances) or cfg["gen_id"]
        cfg = {**cfg, "gen_id": gen_id}
        render = None
        if version == "v18":                  # v18's renderer; older versions' prompt code has since changed
            from ddi import prompt_v18
            render = prompt_v18.render
        specs, raw_found = read_specs(gen_id, render)
        by_sent = group_by_sentence(instances)

        ids = sorted(by_sent)
        chosen = random.Random(seed).sample(ids, min(n_sentences, len(ids)))
        recs = [sentence_record(version, cfg, sid, by_sent[sid], specs)
                for sid in sorted(chosen)]

        path = out_dir / f"{version}.jsonl"
        with open(path, "w") as fp:
            for r in recs:
                fp.write(json.dumps(r) + "\n")

        joined = sum(1 for r in recs if r["spec"] is not None)
        report[version] = {
            **cfg,
            "sentences_available": len(ids),
            "instances_available": len(instances),
            "sentences_sampled": len(recs),
            "raw_file_found": raw_found,
            "specs_joined": joined,
            "prompt_sha": (manifest.get("generator") or {}).get("prompt_sha"),
            "prompt_reproducible_from_repo": prompt_reproducible(version, manifest),
        }
        print(f"{version}: {len(recs)} of {len(ids)} sentences, "
              f"{joined} with specs, raw {'found' if raw_found else 'MISSING'} -> {path}")

    (out_dir / "manifest.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_dir / 'manifest.json'}")


def labelled_sample(out_dir, n_sentences, seed):
    """Real sentences with, for every pair, the gold label, the LLM's label, the verifier's
    verdict on it, and the label after rejected positives are demoted.

    The verifier (2c) was run on the LLM's positive rows only, and build_batches numbers
    mentions over the rows it is given, so verdicts are mapped back to pairs by character
    span using that numbering. The mapping is checked against run_grid.filtered_pool, the
    function the grid and the test run used: the pairs marked rejected here must be exactly
    the pairs whose label that function changed."""
    import run_grid as rg
    H, L, S, _ = rg.load_pools()
    LF, _, _ = rg.filtered_pool(L)

    numbering = {sid: _sentence_spans(rows) for sid, rows in
                 group_by_sentence([r for r in L if r["label"] != "NONE"]).items()}
    verdict = {}
    for v in load_verdicts(rg.VERIFY_GEN).itertuples():
        spans = numbering.get(v.sent_id)
        if spans:
            verdict[(v.sent_id, spans[int(v.m1) - 1], spans[int(v.m2) - 1])] = bool(v.flagged)

    by_sent = defaultdict(list)
    rejected_here, changed_there = set(), set()
    for r, f in zip(L, LF):
        s = _marked_spans(r["text"])
        key = (r["sent_id"], s[0], s[1]) if len(s) == 2 else None
        if r["label"] == "NONE":
            status = "not asked"
        elif key in verdict:
            status = "accepted" if verdict[key] else "rejected"
        else:
            status = "not asked"          # the request for this sentence errored, or nested spans
        if status == "rejected":
            rejected_here.add((r["sent_id"], r["text"]))
        if f["label"] != r["label"]:
            changed_there.add((r["sent_id"], r["text"]))
        by_sent[r["sent_id"]].append({"text": r["text"], "gold": r["gold_label"], "llm": r["label"],
                                      "verdict": status, "final": f["label"]})
    assert rejected_here == changed_there, (
        f"{len(rejected_here)} pairs rejected here, {len(changed_there)} demoted by filtered_pool")

    ids = sorted(by_sent)
    chosen = sorted(random.Random(seed).sample(ids, min(n_sentences, len(ids))))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "labelled.jsonl", "w") as fp:
        for sid in chosen:
            pairs = by_sent[sid]
            rows = [{"sent_id": sid, "text": p["text"], "label": p["gold"]} for p in pairs]
            fp.write(json.dumps({"version": "labelled", "sent_id": sid,
                                 "sentence": _strip(pairs[0]["text"]),
                                 "n_entities": len(_sentence_spans(rows)),
                                 "pairs": pairs, "spec": None}) + "\n")
    counts = defaultdict(int)
    for sid in chosen:
        for p in by_sent[sid]:
            counts[p["verdict"]] += 1
    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man["labelled"] = {"sentences_available": len(ids), "sentences_sampled": len(chosen),
                       "verdicts_in_sample": dict(counts), "verifier_gen_id": rg.VERIFY_GEN,
                       "consistent_with_filtered_pool": True}
    man_path.write_text(json.dumps(man, indent=1))
    print(f"labelled: {len(chosen)} of {len(ids)} sentences, verdicts {dict(counts)} -> "
          f"{out_dir / 'labelled.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tutorial/samples")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labelled", action="store_true",
                    help="write only samples/labelled.jsonl, the real sentences with LLM labels and verdicts")
    a = ap.parse_args()
    if a.labelled:
        labelled_sample(a.out, a.n, a.seed)
    else:
        main(a.out, a.n, a.seed)
