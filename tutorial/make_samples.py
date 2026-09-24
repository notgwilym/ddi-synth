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
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ddi.manifest import load_dataset
from ddi.synth import RAW
from ddi.verify_binary import group_by_sentence, _sentence_spans, _strip

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


def read_specs(gen_id):
    """spec_index -> trimmed spec. Empty dict if the raw file did not survive."""
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
        specs, raw_found = read_specs(cfg["gen_id"])
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tutorial/samples")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.n, a.seed)
