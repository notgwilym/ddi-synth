import json, os, hashlib, subprocess, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# manifests are small -> committed to the repo (like runs/)
MANIFESTS = ROOT / "datasets" / "manifests"
# instances are large & (for synthetic) non-recreatable -> live on NFS, gitignored.
# Set DDI_DATA_ROOT=/root/nfs/ddi_datasets on the pod; falls back to a local dir for testing.
DATA_ROOT = Path(os.environ.get("DDI_DATA_ROOT", ROOT / "datasets" / "instances"))


def _git(*args, default=""):
    # same helper as experiment.py
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def _write_jsonl(path, instances):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in instances:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):   # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


def _label_distribution(instances):
    counts = {}
    for row in instances:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def write_dataset(instances, provenance, *, generator=None, vocab_source=None,
                  negative_strategy=None, seed=None, notes=""):
    """Persist a set of instances + a manifest describing them. Returns the dataset_id.

    instances        : list of {text, label, source, sent_id} dicts
    provenance       : e.g. "human:train", "human:val", "human:test", "synthetic"
    generator        : dict of generation params (model, prompt version, temp...) or None for human
    vocab_source     : {"name": ..., "sha256": ...} for the drug list used, or None
    negative_strategy: label for the Phase-2 ablation axis, or None for now
    """
    dataset_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # 1. persist the instances (filename is always <dataset_id>.jsonl)
    instances_path = DATA_ROOT / f"{dataset_id}.jsonl"
    _write_jsonl(instances_path, instances)

    # 2. build the manifest record
    n_sent = len({row.get("sent_id") for row in instances})
    manifest = {
        "dataset_id": dataset_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provenance": provenance,
        "generator": generator,
        "vocab_source": vocab_source,
        "negative_strategy": negative_strategy,
        "seed": seed,
        "size": {"n_instances": len(instances), "n_sentences": n_sent},
        "label_distribution": _label_distribution(instances),
        "data_root": str(DATA_ROOT),          # informational; load reconstructs from current DATA_ROOT
        "sha256": _sha256(instances_path),    # ties any result back to the exact bytes it used
        "leakage_report": None,               # filled in later by the EM-N gate
        "git_sha": _git("rev-parse", "HEAD", default="NO_GIT"),
        "git_dirty": bool(_git("diff", "HEAD")),
        "notes": notes,
    }

    # 3. write the manifest atomically
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFESTS / f"{dataset_id}.json"
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)

    print(f"wrote dataset {dataset_id}  ({len(instances)} instances, {n_sent} sentences)"
          f"{'  [DIRTY]' if manifest['git_dirty'] else ''}")
    return dataset_id


def load_dataset(dataset_id, verify=True):
    """Load instances + manifest for a dataset_id. Verifies the file hasn't changed."""
    manifest = json.loads((MANIFESTS / f"{dataset_id}.json").read_text())
    path = DATA_ROOT / f"{dataset_id}.jsonl"        # reconstruct from *current* DATA_ROOT
    if verify:
        actual = _sha256(path)
        if actual != manifest["sha256"]:
            raise ValueError(
                f"sha256 mismatch for {dataset_id}: manifest={manifest['sha256'][:12]}, "
                f"file={actual[:12]}. The instances file changed since it was written."
            )
    instances = [json.loads(line) for line in path.read_text().splitlines() if line]
    return instances, manifest