"""Derive an aligned (markers, mask_targets) dataset pair from a pinned marker set.

DDI-2013 has nested and crossing entity spans. Those pairs cannot be masked, so they
are dropped from BOTH members of the pair. The two derived sets therefore contain the
same instances in the same order and differ only in rendering.
"""
import re
from .manifest import load_dataset, write_dataset

_E1 = re.compile(r"\[E1\].*?\[/E1\]", re.S)
_E2 = re.compile(r"\[E2\].*?\[/E2\]", re.S)
_TAGS = re.compile(r"\[/?E[12]\]")

_OK = (["[E1]", "[/E1]", "[E2]", "[/E2]"],
       ["[E2]", "[/E2]", "[E1]", "[/E1]"])


class Unmaskable(Exception):
    """Marker tags are nested or crossing, so the spans overlap."""


def mask_targets_text(text):
    if _TAGS.findall(text) not in _OK:
        raise Unmaskable(text[:200])
    return _E2.sub("drug2", _E1.sub("drug1", text))


def derive_masked_pair(dataset_id, notes=""):
    instances, man = load_dataset(dataset_id)
    kept_markers, kept_masked, dropped = [], [], []
    for r in instances:
        try:
            masked = mask_targets_text(r["text"])
        except Unmaskable:
            dropped.append(r)
            continue
        kept_markers.append(r)
        kept_masked.append({**r, "text": masked})

    from collections import Counter
    print(f"{dataset_id}: {len(dropped)} of {len(instances)} pairs unmaskable "
          f"(overlapping spans), labels {dict(Counter(r['label'] for r in dropped))}")

    common = dict(provenance=man["provenance"], generator=man["generator"],
                  vocab_source=man["vocab_source"],
                  negative_strategy=man["negative_strategy"],
                  seed=man["seed"], parent_id=dataset_id)
    markers_id = write_dataset(kept_markers, render_mode="markers",
                               notes=f"markers, overlap-aligned, from {dataset_id}. {notes}",
                               **common)
    masked_id = write_dataset(kept_masked, render_mode="mask_targets",
                              notes=f"mask_targets, from {dataset_id}. {notes}",
                              **common)
    return markers_id, masked_id, dropped