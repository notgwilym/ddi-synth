"""Label real corpus sentences with an LLM, gold discarded. The labelling arm.

WHY
---
The whole project so far asks whether an LLM can *write* training data. The other way to
spend the same LLM budget is to have it *label* text that already exists. Nobody has run
the two against each other for biomedical relation extraction, and the mixing curve says
the generation arm saturates, so the comparison is the open question rather than another
generator iteration.

Sentences come from held-out Train documents with gold entity spans kept and gold labels
thrown away. That is not the same as labelling raw PubMed: it assumes an NER system, and
so does the generation arm, so the comparison is like for like on the axis that matters.
Say the assumption out loud in the writeup rather than hiding it.

Budget is matched on sentences, not instances, because a sentence is the unit of LLM cost
in both arms. Note that the two arms then differ in pairs per sentence, so also report
instance counts.

reasoning_effort defaults to high and should stay there. At low the five-way verifier's
NONE recall collapsed to 0.45-0.48, prompt changes did not fix it, and effort was the
lever. The v18 smoke run repeated the lesson: at low the binary verifier missed 21% of
true positives on plain synthetic specs against a calibrated 0.948 recall.
"""
import json
import random
from collections import Counter, defaultdict

from .verify_binary import build_batches, render

LABELS = ["MECHANISM", "EFFECT", "ADVISE", "INT", "NONE"]

SYSTEM = """You are annotating drug-drug interactions in biomedical text, following the
DDI-Extraction 2013 guidelines. Every drug mention is numbered inline, like aspirin[3].

For each pair you are asked about, give exactly one label.

  MECHANISM  a pharmacokinetic interaction: one drug changes the absorption,
             distribution, metabolism, excretion, levels, exposure or clearance of the
             other.
  EFFECT     a pharmacodynamic or clinical outcome of the two together: an effect,
             a toxicity, a synergism or antagonism, a loss of efficacy, or protection
             of one against harm from the other.
  ADVISE     a recommendation about using the two together: contraindicated, avoid,
             use with caution, adjust the dose, monitor, separate the doses.
  INT        the text says only that the two interact, with no detail of how.
  NONE       everything else, including drugs merely listed together, denials that any
             interaction occurs, statements about one drug alone, study descriptions
             with no finding reported, and physical incompatibility on mixing.

Rules that decide the hard cases.

Priority: if a pharmacokinetic change is asserted, the label is MECHANISM even when a
clinical consequence is also mentioned. If advice is given and an effect is also stated,
the label is ADVISE.

Both drugs must be participants in the claim. If the sentence asserts something about one
drug and merely mentions the other, the pair is NONE.

An explicit denial is NONE, not a positive label. "No significant change in the
pharmacokinetics of X was seen with Y" is NONE.

A statement that something has not been studied or established is NONE. It is not a
denial that an interaction exists, and it is not a positive.

If the same drug is named twice, only the mention that participates in the claim counts;
pairs involving the other mention are NONE.

Answer for every pair, in the order given."""


def sample_sentences(instances, n=None, seed=0, min_entities=2, max_pairs=190):
    """Sentences, not instances. max_pairs drops the enumeration monsters, matching the
    filter used for the human baseline: eight sentences carry 43% of instances and
    almost no positives, and removing them cost 0.010 F1 (0.800 -> 0.790)."""
    batches = [b for b in build_batches(instances)
               if b["n_mentions"] >= min_entities and len(b["pairs"]) < max_pairs]
    if n is not None:
        rng = random.Random(seed)
        rng.shuffle(batches)
        batches = batches[:n]
    return batches


def chunk(specs, size=8):
    """Split a sentence's pairs across several calls, keeping the whole numbered sentence
    in every one so the model still sees full context.

    At high effort a single call asking about 91 pairs times out or comes back with the
    wrong number of labels: error rate ran 0.60 above 25 pairs, 0.12 at 10-24 and zero
    below four. That cost 26% of all pairs, and because dense sentences are almost all
    NONE the loss was biased -- the kept set ran 0.188 positive against the corpus's
    0.161, so the arm was losing precisely its hardest negatives.

    Note this changes the spec list, and generate_raw resumes on positional index, so a
    chunked run needs a fresh gen-id rather than resuming an unchunked one.
    """
    out = []
    for b in specs:
        n = len(b["pairs"])
        for c, start in enumerate(range(0, n, size)):
            sl = slice(start, start + size)
            out.append({**b, "pairs": b["pairs"][sl], "names": b["names"][sl],
                        "gold": b["gold"][sl], "chunk": c,
                        "n_chunks": (n + size - 1) // size})
    return out


def make_labeller(client, model="gpt-oss-120b", temperature=0.0,
                  reasoning_effort="high", max_output_tokens=8000, api="responses"):
    """max_output_tokens is deliberately large. Reasoning tokens count against the
    budget, and a sentence with 20 pairs at high effort will exceed a small cap and come
    back with output_parsed None, which reads as an API error rather than as truncation.
    """
    from pydantic import BaseModel

    class Labels(BaseModel):
        labels: list[str]

    def _responses(user):
        kw = {"max_output_tokens": max_output_tokens}
        if reasoning_effort:
            kw["reasoning"] = {"effort": reasoning_effort}
        r = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": user}],
            text_format=Labels, temperature=temperature, **kw)
        if r.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(r, 'status', '?')}, "
                             f"raise max_output_tokens if this is truncation)")
        return r.output_parsed.labels

    def _chat(user):
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["labels"],
                  "properties": {"labels": {"type": "array",
                                            "items": {"type": "string",
                                                      "enum": LABELS}}}}
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_output_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "labels", "strict": True, "schema": schema}})
        return Labels.model_validate_json(r.choices[0].message.content).labels

    def label(spec):
        got = (_responses if api == "responses" else _chat)(render(spec))
        if len(got) != len(spec["pairs"]):
            raise ValueError(f"expected {len(spec['pairs'])} labels, got {len(got)}")
        clean = [g if g in LABELS else "NONE" for g in got]
        return {"sent_id": spec["sent_id"], "labels": clean, "gold": spec["gold"],
                "n_pairs": len(spec["pairs"]),
                "chunk": spec.get("chunk", 0), "n_chunks": spec.get("n_chunks", 1),
                "off_schema": sum(1 for g in got if g not in LABELS)}

    return label


def _rows(verdict_path):
    """generate_raw wraps whatever sample_fn returns: {spec_index, spec, sample, error}.
    So the labeller's dict is at ["sample"], and an errored call has sample None."""
    ok, errs = [], []
    for line in open(verdict_path).read().splitlines():
        if not line:
            continue
        r = json.loads(line)
        payload = r.get("sample") if "sample" in r else r
        if r.get("error") or not payload:
            errs.append({"sent_id": (r.get("spec") or {}).get("sent_id"),
                         "n_pairs": len((r.get("spec") or {}).get("pairs") or []),
                         "error": str(r.get("error"))})
        else:
            ok.append(payload)

    # Reassemble chunked sentences. A sentence missing any chunk is dropped whole,
    # because a partial sentence would misalign pair order against the instance list.
    parts = defaultdict(list)
    for o in ok:
        parts[o["sent_id"]].append(o)
    merged, partial = [], 0
    for sent_id, ps in parts.items():
        want = ps[0].get("n_chunks", 1)
        if len(ps) != want:
            partial += 1
            errs.append({"sent_id": sent_id, "n_pairs": sum(p["n_pairs"] for p in ps),
                         "error": f"incomplete: {len(ps)}/{want} chunks"})
            continue
        ps.sort(key=lambda p: p.get("chunk", 0))
        merged.append({"sent_id": sent_id,
                       "labels": [x for p in ps for x in p["labels"]],
                       "gold": [x for p in ps for x in p["gold"]],
                       "n_pairs": sum(p["n_pairs"] for p in ps),
                       "off_schema": sum(p["off_schema"] for p in ps)})
    if partial:
        print(f"{partial} sentences dropped for missing chunks")
    return merged, errs


def _raw_rows(verdict_path):
    """Generic reader for any generate_raw output: {spec_index, spec, sample, error}.
    Makes no assumption about what sample_fn returned, unlike _rows below which is
    specific to the labelling arm's {labels, gold, sent_id} payload. Use this one for
    the generation arm; use error_report(..., kind="label") for the labelling arm."""
    ok, errs = [], []
    for line in open(verdict_path).read().splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r.get("error") or not r.get("sample"):
            errs.append({"spec": r.get("spec") or {}, "error": str(r.get("error"))})
        else:
            ok.append({"spec": r.get("spec") or {}, "sample": r["sample"]})
    return ok, errs


def error_report(verdict_path, kind="auto"):
    """Where the failures are. At high effort they concentrate in specs with many
    pairs or assertions, because reasoning tokens scale with the number of things
    being asked about and both the token cap and the client timeout bite there first.

    kind="label" reads the labelling arm's {sent_id, labels, gold} payload and buckets
    by pairs per call, which is what caught the label-high run's chunking need.
    kind="generate" reads any generate_raw output generically and buckets by whatever
    size signal is in the spec: n_pairs if present, else n_assertions, else entity
    count, else just reports the flat error rate.
    kind="auto" (default) tries "label" first and falls back to "generate" if the
    payload doesn't have the expected keys, so this one function works on either
    arm's raw file without the caller needing to know which arm produced it.
    """
    if kind == "auto":
        # Peek at one successful record's payload shape before choosing a parser.
        # _rows assumes {sent_id, labels, gold} and does chunk reassembly keyed on
        # sent_id; running it on a generation file (payload = {"sentence": ...}, no
        # sent_id) throws partway through, so check the shape first rather than
        # catching the crash after parsing however many lines came before it.
        kind = "generate"
        for line in open(verdict_path).read().splitlines():
            if not line:
                continue
            r = json.loads(line)
            payload = r.get("sample") if "sample" in r else r
            if not r.get("error") and payload:
                kind = "label" if "sent_id" in payload and "labels" in payload \
                    else "generate"
                break
        ok, errs = (_rows(verdict_path) if kind == "label"
                   else _raw_rows(verdict_path))
    elif kind == "label":
        ok, errs = _rows(verdict_path)
    else:
        ok, errs = _raw_rows(verdict_path)

    print(f"{len(ok)} ok, {len(errs)} errored, "
          f"{len(errs) / max(len(ok) + len(errs), 1):.3f} error rate")
    if not errs:
        return
    print("\nerror kinds")
    for c, n in Counter(e["error"][:70] for e in errs).most_common(6):
        print(f"  {n:5d}  {c}")

    def bucket(n):
        return "1" if n <= 1 else "2-3" if n <= 3 else "4-9" if n <= 9 else \
               "10-24" if n <= 24 else "25+"

    if kind == "label":
        tot = Counter(bucket(len(o["labels"])) for o in ok)
        bad = Counter(bucket(e["n_pairs"]) for e in errs)
        label_col = "pairs"
    else:
        def size(spec):
            if "n_pairs" in spec:
                return spec["n_pairs"]
            if "assertions" in spec:
                return len(spec["assertions"])
            if "entities" in spec:
                return len(spec["entities"])
            return -1

        sizes_ok = [size(o["spec"]) for o in ok]
        sizes_err = [size(e["spec"]) for e in errs]
        if all(s == -1 for s in sizes_ok + sizes_err):
            print("\nno size signal found in spec (no n_pairs/assertions/entities); "
                  "can't bucket, only the flat rate above is available")
            return
        label_col = ("assertions" if "assertions" in ok[0]["spec"] else
                    "entities" if ok else "size")
        tot = Counter(bucket(s) for s in sizes_ok)
        bad = Counter(bucket(s) for s in sizes_err)

    print(f"\n{label_col:>7} {'ok':>6} {'err':>6} {'err rate':>9}")
    for b in ["1", "2-3", "4-9", "10-24", "25+"]:
        t, e = tot[b], bad[b]
        print(f"{b:>7} {t:>6} {e:>6} {e / max(t + e, 1):>9.3f}")


def to_instances(instances, verdict_path):
    """Rebuild the instance list with LLM labels in place of gold. Sentences that errored
    are dropped whole; a partially labelled sentence would misalign pair order."""
    ok, _ = _rows(verdict_path)
    got = {o["sent_id"]: o["labels"] for o in ok}

    spec_pairs = {b["sent_id"]: b for b in build_batches(instances)}
    out, dropped = [], 0
    for sent_id, labels in got.items():
        b = spec_pairs.get(sent_id)
        if b is None or len(labels) != len(b["pairs"]):
            dropped += 1
            continue
        rows = [r for r in instances if r["sent_id"] == sent_id]
        keep = [r for r in rows if len(_two(r)) == 2]
        if len(keep) != len(labels):
            dropped += 1
            continue
        for r, lab in zip(keep, labels):
            out.append({**r, "label": lab, "gold_label": r["label"]})
    return out, dropped


def _two(r):
    from .verify_binary import _marked_spans
    return _marked_spans(r["text"])


def agreement(verdict_path):
    """How close the LLM labels are to gold. Not the point of the experiment, but it is
    the number that tells you whether a low downstream F1 means bad labels or something
    else, so measure it."""
    from sklearn.metrics import classification_report
    ok, errs = _rows(verdict_path)
    print(f"{len(ok)} sentences labelled, {len(errs)} errored\n")
    y, yhat = [], []
    for o in ok:
        if len(o["gold"]) != len(o["labels"]):
            continue
        y += list(o["gold"])
        yhat += list(o["labels"])
    print(classification_report(y, yhat, digits=3, zero_division=0))
    pos = [(a, b) for a, b in zip(y, yhat) if a != "NONE" or b != "NONE"]
    ok = sum(1 for a, b in pos if a == b)
    pos = [c for c in LABELS if c != "NONE"]
    tp = sum(1 for a, b in zip(y, yhat) if a == b and a != "NONE")
    n_gold = sum(1 for a in y if a != "NONE")
    n_pred = sum(1 for b in yhat if b != "NONE")
    mp, mr = tp / max(n_pred, 1), tp / max(n_gold, 1)
    print(f"micro over positive classes:  P {mp:.3f}  R {mr:.3f}  "
          f"F1 {2 * mp * mr / max(mp + mr, 1e-9):.3f}")
    print("reference: human-trained 0.790, v18 generator 0.486\n")
    print(f"pairs {len(y)}, gold positives {sum(1 for a in y if a != 'NONE')}, "
          f"llm positives {sum(1 for b in yhat if b != 'NONE')}")
    print(f"agreement over pairs either side calls positive: {ok / max(len(pos), 1):.3f}")
    print(Counter(yhat).most_common())