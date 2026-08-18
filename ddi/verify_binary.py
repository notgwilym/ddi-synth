"""Binary assertion verifier, positionally indexed.

WHY THE FIRST VERSION FAILED
----------------------------
NONE recall 0.532 excluding same-entity pairs, against the five-way verifier's 0.81.
Reading the false positives showed the verifier was mostly right and the questions were
unanswerable.

"When concomitant administration of ketoconazole with aripiprazole occurs, aripiprazole
dose should be reduced to one-half of its normal dose" contains two aripiprazole
mentions. DDI-2013 rule P1 says only the mention most logically linked to the textual
evidence participates, so ketoconazole-aripiprazole(first) is ADVISE and
ketoconazole-aripiprazole(second) is NONE. The old batcher named pairs by surface form,
so both rendered as "ketoconazole / aripiprazole": the same question twice, in the same
sentence, with two different correct answers.

The fix is to number every entity mention inline in the sentence and refer to pairs by
index. Only then do the pair-selection rules have anything to select between.

The other systematic miss was negation. "ZINECARD does not influence the
pharmacokinetics of doxorubicin" is gold NONE because DDI-2013 does not annotate negated
interactions. That is a convention, not something a careful reader derives, and the
first prompt did not state it.

Batch size was checked and is not a factor: NONE recall is flat across batch sizes, and
the mean batch is 4.3 pairs. Coordination was checked and is inverted from what I
predicted: coordinated pairs score 0.812, ordinary co-mentions 0.464.

WHAT THIS IS FOR
----------------
Measuring, not filtering. v15 trained 0.046 worse than v14 while being closer to the
corpus on every divergence measurement, and composition-matched arms did not recover it.
Label accuracy is the remaining unmeasured variable. A verifier is the only instrument
for it, and it is only usable if its false-positive rate on true NONEs is low enough
that the flags are not swamped: NONE recall near 0.95 for a 3% base rate.
"""
import json
import re
from collections import defaultdict

from .synth import RAW

MARKER = re.compile(r"\[/?E[12]\]")


SYSTEM = """You are annotating drug-drug interactions in biomedical text, following the
DDI-Extraction 2013 guidelines.

The sentence has every drug mention numbered inline, like this: aspirin[3]. Numbers mark
mentions, not substances: the same drug named twice gets two numbers, and they are
different mentions.

For each numbered pair you are given, decide whether the guidelines would annotate a
relation between those two specific mentions. Answer true or false.

WHAT COUNTS AS AN INTERACTION

A drug-drug interaction is a change in the effects of one drug caused by the presence of
another. Annotate a pair if the sentence asserts any of these about it:

- a pharmacokinetic change: absorption, distribution, metabolism, excretion, or a change
  in levels, concentration, AUC, Cmax, bioavailability, clearance, half-life, protein
  binding, or enzyme induction or inhibition
- a pharmacological effect, clinical finding, toxicity, therapeutic failure, protective
  effect, or a pharmacodynamic relation such as synergism, antagonism or potentiation
- a recommendation about using them together: contraindicated, avoid, use with caution,
  adjust the dose, monitor
- a bare statement that they interact, with no detail
- When a substance is mentioned more than once, only the mention most closely tied to
  the interaction takes part. That is normally the mention inside the clause naming both
  drugs together, not a later mention in a clause about dosing or consequence. In "when
  carbamazepine[1] is added to aripiprazole[2] therapy, aripiprazole[3] dose should be
  doubled", the pair is 1 and 2, not 1 and 3. Mentions inside "coadministration of X and
  Y", "concomitant administration of X and Y", or "when X was given with Y" are normally
  the participating ones.

Hedged and uncertain interactions still count. "May", "possible", "suggested" and
"predicted" are annotated. Beneficial interactions count too; direction of benefit is
irrelevant.

WHAT DOES NOT COUNT

- Negated interactions. If the sentence says there is no interaction, no effect, or that
  something is unaffected, the pair is not annotated. No exception.
- A study that was performed without a result being reported. "The interaction of X and
  Y was studied" asserts only that an investigation happened.
- Two drugs merely administered together, listed together, or compared, with nothing
  said about one changing the other. This is the most common case.
- Incompatibility outside the body: precipitation, formulation chemistry, a drug
  reacting with an excipient or container.
- A claim about one of them and some third drug. If the sentence says drug[1] raises the
  levels of drug[3], then the pair 1 and 3 is annotated, but 1 and 2 is not, even when
  drug[2] sits in the same clause.

If the sentence both affirms and denies an interaction, the affirmation wins.

WHICH MENTIONS

- Two mentions of the same substance never interact with each other.
- When a substance is mentioned more than once, only the mention most closely tied to
  the interaction takes part. Mentions inside phrases like "coadministration of X and
  Y", "concomitant administration of X and Y", or "when X was given with Y" are normally
  the participating ones. Other mentions of the same substance elsewhere in the sentence
  are not.
- In a sentence of the form "Entity: <statement>", the leading entity participates only
  if it is the subject of the statement. If the statement names its own pair, the
  heading mention does not take part.
- If a class and a member of that class are both named, and the assertion ranges over
  both, annotate both sides. "Quinolones, including cinoxacin, may enhance the effects
  of oral anticoagulants, such as warfarin" gives four relations.
- A sentence can license several relations, over the same pair or different ones. Having
  assigned an interaction to one pair is not a reason to withhold it from another.

Judge each pair independently. Return JSON: one entry per pair, {"pair": <index>,
"annotated": true or false}. No other keys, no commentary."""


def _strip(text):
    return MARKER.sub("", text)


def group_by_sentence(instances):
    """Sentence -> its pair instances. Every instance from one sentence is the same text
    with different markers."""
    by_sent = defaultdict(list)
    for r in instances:
        by_sent[r["sent_id"]].append(r)
    return by_sent


def _marked_spans(text):
    out, shift, start = [], 0, None
    for m in re.finditer(r"\[(/?)E([12])\]", text):
        if m.group(1) == "":
            start = m.start() - shift
            shift += len(m.group(0))
        else:
            out.append((start, m.start() - shift))
            shift += len(m.group(0))
    return sorted(out)


def _sentence_spans(rows):
    """Every distinct mention span in the sentence, from all its pair instances."""
    spans = set()
    for r in rows:
        for s in _marked_spans(r["text"]):
            spans.add(s)
    return sorted(spans)


def _number_inline(text, spans):
    """aspirin -> aspirin[3], inserted right to left so offsets stay valid."""
    out = text
    for i, (b, e) in reversed(list(enumerate(spans, start=1))):
        out = out[:e] + f"[{i}]" + out[e:]
    return out


def build_batches(instances):
    """One spec per sentence. Pairs are (index, index) into the numbered mentions, not
    surface forms, so a repeated drug name produces distinguishable questions."""
    batches = []
    for sent_id, rows in group_by_sentence(instances).items():
        plain = _strip(rows[0]["text"])
        spans = _sentence_spans(rows)
        idx = {s: i for i, s in enumerate(spans, start=1)}

        pairs, gold, names = [], [], []
        for r in rows:
            s = _marked_spans(r["text"])
            if len(s) != 2 or any(x not in idx for x in s):
                continue                        # nested or overlapping spans
            pairs.append([idx[s[0]], idx[s[1]]])
            gold.append(r["label"])
            names.append([plain[s[0][0]:s[0][1]], plain[s[1][0]:s[1][1]]])
        if not pairs:
            continue
        batches.append({"sent_id": sent_id,
                        "text": _number_inline(plain, spans),
                        "pairs": pairs, "names": names, "gold": gold,
                        "n_mentions": len(spans)})
    return batches


def render(spec):
    lines = [spec["text"], "", "pairs:"]
    for i, (p, nm) in enumerate(zip(spec["pairs"], spec["names"])):
        lines.append(f"  {i}. mention {p[0]} ({nm[0]})  and  mention {p[1]} ({nm[1]})")
    return "\n".join(lines)


def make_binary_verifier(client, model="gpt-oss-120b", temperature=0.0,
                         reasoning_effort="high", max_output_tokens=4000,
                         api="responses"):
    """reasoning_effort=high is not optional. At low the five-way verifier's NONE recall
    collapsed to 0.45-0.48 and prompt changes did not fix it; effort was the lever."""
    from pydantic import BaseModel

    class Verdict(BaseModel):
        pair: int
        annotated: bool

    class Verdicts(BaseModel):
        verdicts: list[Verdict]

    def _responses(user):
        kw = {}
        if reasoning_effort:
            kw["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens:
            kw["max_output_tokens"] = max_output_tokens
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": user}],
            text_format=Verdicts, temperature=temperature, **kw)
        if resp.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(resp, 'status', '?')})")
        return resp.output_parsed.model_dump()

    def _chat(user):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_output_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "verdicts", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "required": ["verdicts"],
                           "properties": {"verdicts": {"type": "array", "items": {
                               "type": "object", "additionalProperties": False,
                               "required": ["pair", "annotated"],
                               "properties": {"pair": {"type": "integer"},
                                              "annotated": {"type": "boolean"}}}}}}}})
        return Verdicts.model_validate_json(
            resp.choices[0].message.content).model_dump()

    def verify_fn(spec):
        return (_responses if api == "responses" else _chat)(render(spec))

    return verify_fn


def load_verdicts(gen_id):
    """Flatten a batched run back to one row per pair. A batch returning the wrong
    number of verdicts is dropped whole, since index misalignment would silently
    corrupt the measurement."""
    import pandas as pd

    rows, n_err, n_short = [], 0, 0
    for line in (RAW / f"{gen_id}.jsonl").read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("error") or not rec.get("sample"):
            n_err += 1
            continue
        spec = rec["spec"]
        got = {v["pair"]: v["annotated"] for v in rec["sample"]["verdicts"]}
        if len(got) != len(spec["pairs"]):
            n_short += 1
            continue
        for i, (pair, nm, gold) in enumerate(
                zip(spec["pairs"], spec["names"], spec["gold"])):
            rows.append({"sent_id": spec["sent_id"], "text": spec["text"],
                         "m1": pair[0], "m2": pair[1], "e1": nm[0], "e2": nm[1],
                         "n_mentions": spec["n_mentions"],
                         "gold": gold, "gold_pos": gold != "NONE",
                         "flagged": bool(got.get(i, False))})
    if n_err or n_short:
        print(f"dropped {n_err} errored batches, {n_short} with mismatched counts")
    return pd.DataFrame(rows)


def calibration_report(df, base_rate=0.03, n_none=31000):
    """NONE recall is the number that decides everything. True NONEs outnumber drifted
    pairs by roughly thirty to one in the synthetic data, so a small false-positive rate
    on the majority class produces more flags than the entire population being sought."""
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(
        df.gold_pos.values, df.flagged.values, labels=[False, True]).ravel()
    none_recall = tn / max(tn + fp, 1)
    pos_recall = tp / max(tp + fn, 1)

    print(f"n = {len(df)}")
    print(f"  NONE recall     {none_recall:.3f}   [{tn}/{tn + fp}]")
    print(f"  positive recall {pos_recall:.3f}   [{tp}/{tp + fn}]")
    print(f"  accuracy        {(tn + tp) / len(df):.3f}")

    fpr = 1 - none_recall
    n_drift = n_none * base_rate
    false_flags = (n_none - n_drift) * fpr
    true_flags = n_drift * pos_recall
    prec = true_flags / max(false_flags + true_flags, 1)
    print(f"\nprojected onto ~{n_none} synthetic NONE pairs at {base_rate:.0%} drift:")
    print(f"  false {false_flags:>7.0f}  true {true_flags:>6.0f}  "
          f"flag precision {prec:.3f}")
    print("  usable as a measurement above ~0.3, which needs NONE recall near 0.95")
    return {"none_recall": none_recall, "pos_recall": pos_recall,
            "flag_precision": prec}