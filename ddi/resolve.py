"""Stage-2 resolver, plus the diagnostics the gates need.

Labels come from the spec, not from model output, because asking for a label back means
putting label vocabulary in the prompt. build_dataset_from_raw already has the spec in
hand: it reads rec["spec"] for the register.

Handles both spec shapes. v14/v15/v16 put asserted pairs under "asserts"; v17 frames put
them under "positives". Everything else downstream is identical.

Matching is deliberately tolerant. _find_nth already falls back to case-insensitive, and
every rejected sample is a wasted generation call, which is 90% of the loop. Rejects are
a diagnostic for prompt iteration, not a quality filter: the label has to be right by
construction rather than by discarding pairs afterwards.
"""
import json

from .synth import (RAW, Rejected, SynthDoc, SynthRelation, _normalise,
                    _is_degenerate, _entities_in_sentence,
                    _make_pair_instances_synth)


def _asserted(spec):
    """v17 calls them positives, earlier versions called them asserts. Same shape:
    {"between": [key, key], "label": str}."""
    return spec.get("asserts") or spec.get("positives") or []


def v14_sample_to_instances(sample, sent_id_base, register="synthetic", spec=None,
                            max_words=120, mode="markers"):
    if spec is None:
        raise Rejected("resolver needs the spec")
    if not isinstance(sample, dict):
        raise Rejected("no sample returned")

    text = _normalise(sample.get("sentence") or "")
    if not text:
        raise Rejected("empty sentence")
    if len(text.split()) > max_words:
        raise Rejected(f"sentence too long ({len(text.split())} words)")
    if _is_degenerate(text):
        raise Rejected("degenerate repetition")

    surfaces = [e["surface"] for e in spec["entities"]]
    ents, name_of = _entities_in_sentence(text, surfaces)

    found = {name_of[id(e)] for e in ents}
    missing = [s for s in surfaces if s.lower() not in found]
    if missing:
        raise Rejected(f"drug not in sentence: {missing[0]}")

    by_key = {e["key"]: e["surface"].lower() for e in spec["entities"]}
    rels = []
    for asrt in _asserted(spec):
        k1, k2 = asrt["between"]
        m1 = [e for e in ents if name_of[id(e)] == by_key[k1]]
        m2 = [e for e in ents if name_of[id(e)] == by_key[k2]]
        if not m1 or not m2:
            raise Rejected(f"asserted pair {by_key[k1]}~{by_key[k2]} incomplete")
        best = min(((e1, e2) for e1 in m1 for e2 in m2),
                   key=lambda p: abs(p[0].locations.begin() - p[1].locations.begin()))
        rels.append(SynthRelation(best[0].id, best[1].id, asrt["label"]))

    doc = SynthDoc(text, ents, rels, register, sent_id_base)
    instances, self_binds = _make_pair_instances_synth(doc, mode=mode)
    if not instances:
        raise Rejected("no pairs produced")
    if self_binds:
        print(f"warning: {self_binds} self-relations ignored in {sent_id_base}")
    return instances


def _iter_raw(gen_id):
    for line in (RAW / f"{gen_id}.jsonl").read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("error") or not rec.get("sample"):
            continue
        yield rec["spec"], _normalise(rec["sample"].get("sentence") or "")


def generation_records(gen_id):
    """Per-sentence records for the gates: where role-bearing drugs sit relative to the
    asserted pair, and the span actually achieved between the pair.

    v17 specs carry a frame and no role_pos, so those fields come back None. The
    role-position gate then computes over an empty list and returns 0.0, which is
    correct: there is no requested position to have been ignored.
    """
    out = []
    for spec, text in _iter_raw(gen_id):
        if not text:
            continue
        low = text.lower()
        by_key = {e["key"]: e["surface"].lower() for e in spec["entities"]}
        pos = {k: low.find(s) for k, s in by_key.items()}
        asserts = _asserted(spec)

        rec = {"sentence": text, "register": spec["register"],
               "frame": spec.get("frame"),
               "label": asserts[0]["label"] if asserts else "NONE",
               "n_entities": len(spec["entities"]),
               "role_groups": [],
               "role_pos_requested": spec.get("role_pos"),
               "positive_span": None, "role_spans": [], "sep_achieved": None}

        if asserts:
            a, b = asserts[0]["between"]
            if pos.get(a, -1) >= 0 and pos.get(b, -1) >= 0:
                lo, hi = sorted((pos[a], pos[b]))
                rec["positive_span"] = (lo + hi) / 2
                # separation is no longer requested; measured anyway, since it is the
                # span distribution the classifier actually sees
                rec["sep_achieved"] = sum(1 for k, p in pos.items()
                                          if k not in (a, b) and lo < p < hi)
        rec["role_spans"] = [pos[k] for k in spec.get("roles", {})
                             if pos.get(k, -1) >= 0]
        out.append(rec)
    return out


def role_position_report(records):
    """Did the model put the non-participants where it was told? Left free in v14 they
    landed after the assertion 96% of the time, which lets a classifier read clause
    position instead of the entity markers. v17 does not request a position, so this
    reports nothing for it."""
    from collections import Counter, defaultdict
    got = defaultdict(Counter)
    for r in records:
        req, anchor = r.get("role_pos_requested"), r.get("positive_span")
        if not req or anchor is None:
            continue
        for s in r.get("role_spans", []):
            got[req]["before" if s < anchor else "after"] += 1
    if not got:
        print("no role positions requested")
        return
    print(f"{'requested':>10}  {'n':>5}  achieved")
    for req in sorted(got):
        row = got[req]
        n = sum(row.values())
        print(f"{req:>10}  {n:>5}  " +
              "  ".join(f"{k}:{v / n:.2f}" for k, v in sorted(row.items())))


def separation_report(records):
    """Other drug names falling between the asserted pair. Real positives have a median
    token gap of 10 and p90 of 57, so a distribution pinned at 0 means synthetic
    positives are all adjacent and long-span positives are unrepresented."""
    from collections import Counter
    got = Counter(r["sep_achieved"] for r in records if r["sep_achieved"] is not None)
    n = sum(got.values())
    if not n:
        return
    print("names between the asserted pair:",
          "  ".join(f"{k}:{v / n:.2f}" for k, v in sorted(got.items())))