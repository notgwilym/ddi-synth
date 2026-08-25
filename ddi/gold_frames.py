"""Classify real corpus sentences into v17's frame inventory.

WHY
---
Two questions the synthetic side cannot answer about itself.

Coverage: which constructions the corpus uses that no frame writes. The frame inventory
was built from the annotation guidelines' worked examples, so it covers what the
guidelines chose to illustrate, which is not the same as what DrugBank and MedLine
sentences actually do. Anything landing in `other` is a candidate frame, and the
descriptions cluster into the ones worth adding.

Purity: what share of the pairs in a real sentence carrying a given construction are
positive. v17's frames are 0.000 or 1.000 by construction. If real `denial` sentences
run at, say, 0.08 rather than 0.000, the target for v18 is that number, not zero, and
per-pair construction sampling is the way to hit it.

Gold labels are deliberately withheld from the classifier. Showing them would let it
infer the construction from the label, which manufactures exactly the purity the
measurement is meant to test.

A few hundred sentences is enough for both. Frame shares below about 0.02 will not be
estimated usefully at n=300, which is acceptable: the frames that matter for weighting
are the large ones, and the small ones are diagnosed from the `other` bucket instead.
"""
import json
import random
from collections import Counter, defaultdict

from .verify_binary import _number_inline, _sentence_spans, _strip, group_by_sentence

FRAME_DESCRIPTIONS = {
    "regimen": "drugs named as parts of a treatment plan, alternatives for the same "
               "indication, or items in a list, with no claim that one affects another",
    "denial": "an explicit statement that no change, no effect or no interaction occurs",
    "study_only": "a study design, dosing or administration statement with no outcome "
                  "claimed",
    "incompatible": "physical or chemical incompatibility when two products are mixed "
                    "before administration",
    "enzyme_only": "a drug described as an inducer, inhibitor or substrate of an enzyme "
                   "or transporter, without a second drug named as affected",
    "out_of_scope": "a change described in a process, measure or non-drug entity rather "
                    "than in one of the named drugs",
    "sequential": "drugs given at separate times, with a washout, gap or stop-then-start "
                  "structure",
    "mechanism": "one drug changing a pharmacokinetic quantity of another: absorption, "
                 "distribution, metabolism, excretion, levels, exposure or clearance",
    "mechanism_mixed": "a pharmacokinetic change followed by a clinical consequence in "
                       "the same sentence",
    "effect": "a clinical or pharmacological outcome arising from taking two drugs "
              "together",
    "effect_pd": "pharmacodynamic language: synergism, antagonism, potentiation, "
                 "additive, blunted or enhanced response",
    "effect_protect": "one drug protecting against or reducing harm caused by another",
    "effect_failure": "loss of efficacy, diminished benefit or therapeutic failure",
    "advise": "a recommendation about combined use: contraindicated, avoid, use with "
              "caution, adjust the dose, monitor",
    "advise_reason": "a recommendation with the reason for it stated in the same "
                     "sentence",
    "int": "a bare statement that two drugs interact, with no detail of how",
    "contradictory": "a denial followed by an affirmation that overrides it, or the "
                     "reverse",
    "title": "a heading or title-like fragment naming a drug, followed by the statement",
    "coordinate": "two or more coordinated drugs jointly acting on a further drug",
    "appositive": "a class named with a member, such as X including Y or X such as Y, "
                  "where the statement holds across both",
}

SYSTEM = """You are identifying the rhetorical construction of sentences from biomedical
drug labels and research abstracts. Every drug mention is numbered inline, like
aspirin[3].

Your job is to name the construction the sentence uses, not to decide whether a drug
interaction is present. Report what the sentence does, regardless of whether it asserts
an interaction. Do not reason about annotation guidelines.

The constructions:

{catalogue}

Rules.

Choose exactly one `primary`: the construction that carries the sentence's main clause.

List any others genuinely present in `also_present`. Real sentences often do more than
one thing at once, for example a recommendation attached to a mechanism, or a denial
about one pair inside a sentence that asserts an effect about another. Report all of
them. Leave the list empty if the sentence does one thing.

If no construction in the catalogue fits the main clause, set `primary` to `other` and
write a one-line description of the construction in `other_description`, in the same
style as the catalogue entries. Describe the construction, not the sentence's content.
Prefer `other` over a poor fit: a wrong forced match is worse than an honest gap.

Set `confident` to false if the sentence is truncated, is a bare list of drug names with
no predicate, or is otherwise not classifiable."""


def catalogue():
    return "\n".join(f"  {k}: {v}" for k, v in FRAME_DESCRIPTIONS.items())


def system_prompt():
    return SYSTEM.format(catalogue=catalogue())


def sample_sentences(instances, n=300, seed=0, min_entities=2):
    """Stratified by source and by whether the sentence carries any positive pair, so
    the negative constructions are not swamped. The corpus is 84% negative sentences and
    an unstratified draw would spend most of the budget on `regimen`."""
    by_sent = group_by_sentence(instances)
    pool = []
    for sent_id, rows in by_sent.items():
        spans = _sentence_spans(rows)
        if len(spans) < min_entities:
            continue
        has_pos = any(r["label"] != "NONE" for r in rows)
        source = rows[0].get("source") or rows[0].get("register") or "?"
        pool.append({"sent_id": sent_id, "rows": rows, "spans": spans,
                     "has_pos": has_pos, "source": source})

    strata = defaultdict(list)
    for p in pool:
        strata[(p["source"], p["has_pos"])].append(p)

    rng = random.Random(seed)
    per = max(1, n // max(1, len(strata)))
    out = []
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        out.extend(group[:per])
    rng.shuffle(out)
    return out[:n]


def render(item):
    plain = _strip(item["rows"][0]["text"])
    return _number_inline(plain, item["spans"])


def make_frame_classifier(client, model="gpt-oss-120b", temperature=0.0,
                          reasoning_effort="high", max_output_tokens=3000,
                          api="responses"):
    """reasoning_effort follows the binary verifier: at low the five-way verifier's
    recall collapsed and effort was the lever, and this task is harder than that one."""
    from pydantic import BaseModel

    class FrameVerdict(BaseModel):
        primary: str
        also_present: list[str]
        other_description: str
        confident: bool

    system = system_prompt()

    def _responses(user):
        kw = {}
        if reasoning_effort:
            kw["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens:
            kw["max_output_tokens"] = max_output_tokens
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
            text_format=FrameVerdict, temperature=temperature, **kw)
        if resp.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(resp, 'status', '?')})")
        return resp.output_parsed.model_dump()

    def _chat(user):
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["primary", "also_present", "other_description",
                               "confident"],
                  "properties": {
                      "primary": {"type": "string"},
                      "also_present": {"type": "array",
                                       "items": {"type": "string"}},
                      "other_description": {"type": "string"},
                      "confident": {"type": "boolean"}}}
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_output_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "frame_verdict", "strict": True, "schema": schema}})
        return FrameVerdict.model_validate_json(
            resp.choices[0].message.content).model_dump()

    def classify(item):
        return (_responses if api == "responses" else _chat)(render(item))

    return classify


def run(items, classify, out_path, workers=8):
    """Appends one json object per sentence. Resumes from whatever is already on disk,
    matching generate_raw, so an interrupted run costs nothing."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line:
                done.add(json.loads(line)["sent_id"])

    todo = [i for i in items if i["sent_id"] not in done]
    if not todo:
        return len(done), 0

    errors = 0
    with out_path.open("a") as fh, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(classify, item): item for item in todo}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                verdict = fut.result()
            except Exception as exc:
                errors += 1
                verdict = {"error": str(exc)}
            row = {"sent_id": item["sent_id"], "source": item["source"],
                   "has_pos": item["has_pos"], "text": render(item),
                   "n_pairs": len(item["rows"]), **verdict}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    return len(done) + len(todo) - errors, errors


def load(path):
    rows = []
    for line in open(path).read().splitlines():
        if line:
            rows.append(json.loads(line))
    return [r for r in rows if not r.get("error")]


def corpus_distribution(verdicts, include_secondary=False):
    """Share of corpus sentences per construction. With include_secondary the count is
    over all constructions present, which sums above 1 and is the right denominator for
    asking whether a frame should exist at all."""
    c = Counter()
    for v in verdicts:
        if not v.get("confident", True):
            continue
        c[v["primary"]] += 1
        if include_secondary:
            for f in v.get("also_present") or []:
                if f != v["primary"]:
                    c[f] += 1
    total = sum(c.values()) or 1
    return {k: n / total for k, n in c.most_common()}, c


def corpus_purity(verdicts, instances, cap=20):
    """Per construction: how often a real sentence carrying it has any positive pair.

    Reports three numbers because the obvious one is broken. Pair-level positive rate
    over every pair in every sentence carrying a construction is dominated by a handful
    of enormous sentences: one Anisindione sentence contributes 1081 of the 1339 pairs
    in the `effect` bucket, and one Chlorpromazine sentence contributes 105 of the 131
    in `int`, so both those figures are one sentence each.

    `sent_pos_rate` is the number to aim at. It is P(sentence has at least one positive |
    construction present), which is the direct analogue of frame_diag.label_purity's
    nmi_sentence, currently 1.000 on v17.
    """
    by_sent = group_by_sentence(instances)
    sent = defaultdict(Counter)
    pairs = defaultdict(Counter)
    capped = defaultdict(Counter)
    for v in verdicts:
        rows = by_sent.get(v["sent_id"]) or []
        if not rows:
            continue
        has_pos = any(r["label"] != "NONE" for r in rows)
        for f in {v["primary"], *(v.get("also_present") or [])}:
            sent[f]["POS" if has_pos else "NONE"] += 1
            for r in rows:
                lab = "POS" if r["label"] != "NONE" else "NONE"
                pairs[f][lab] += 1
                if len(rows) <= cap:
                    capped[f][lab] += 1

    out = []
    for f, c in sent.items():
        n_s = c["POS"] + c["NONE"]
        p, q = pairs[f], capped[f]
        n_p, n_c = p["POS"] + p["NONE"], q["POS"] + q["NONE"]
        out.append({"frame": f, "n_sent": n_s, "n_pairs": n_p,
                    "sent_pos_rate": c["POS"] / n_s if n_s else 0.0,
                    "pair_pos_rate": p["POS"] / n_p if n_p else 0.0,
                    "pair_pos_rate_capped": q["POS"] / n_c if n_c else 0.0,
                    "n_pairs_capped": n_c})
    out.sort(key=lambda r: -r["n_sent"])
    return out


def multi_construction_rate(verdicts):
    """How often a real sentence does more than one thing. v17 sentences do exactly one
    by construction, so any gap here is a structural divergence the divergence framework
    does not currently measure."""
    ok = [v for v in verdicts if v.get("confident", True)]
    if not ok:
        return {}
    n_multi = sum(1 for v in ok if len(v.get("also_present") or []) > 0)
    mixed = sum(1 for v in ok
                if v["has_pos"] and any(f in NEGATIVE_FRAMES
                                        for f in (v.get("also_present") or [])))
    return {"n": len(ok), "multi_rate": n_multi / len(ok),
            "mean_constructions": sum(1 + len(v.get("also_present") or [])
                                      for v in ok) / len(ok),
            "positive_sentences_carrying_a_negative_construction":
                mixed / max(1, sum(1 for v in ok if v["has_pos"]))}


NEGATIVE_FRAMES = {"regimen", "denial", "study_only", "incompatible", "enzyme_only",
                   "out_of_scope", "sequential"}


PROPOSE_SYSTEM = """You are given short descriptions of sentence constructions found in
biomedical drug-interaction text that did not match an existing catalogue.

Group them into a small number of distinct constructions. Merge descriptions that name
the same thing in different words. Discard any that are restatements of a catalogue entry
rather than something new.

For each group, give a short lowercase snake_case name, a one-line description in the
style of the catalogue, the number of input descriptions it covers, and one representative
input description verbatim.

The existing catalogue is:

{catalogue}"""


def propose_frames(client, verdicts, model="gpt-oss-120b", reasoning_effort="high",
                   max_output_tokens=4000):
    """Second pass over the `other` descriptions. Grouping by hand works too and is
    worth doing on a sample regardless, since the groups are the candidate frames and
    reading them is how the inventory gets fixed."""
    from pydantic import BaseModel

    class Group(BaseModel):
        name: str
        description: str
        n: int
        example: str

    class Groups(BaseModel):
        groups: list[Group]

    others = [v["other_description"] for v in verdicts
              if v.get("primary") == "other" and v.get("other_description")]
    if not others:
        return []

    resp = client.responses.parse(
        model=model,
        input=[{"role": "system",
                "content": PROPOSE_SYSTEM.format(catalogue=catalogue())},
               {"role": "user", "content": "\n".join(f"- {d}" for d in others)}],
        text_format=Groups, temperature=0.0,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens)
    return resp.output_parsed.model_dump()["groups"]