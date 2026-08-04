"""Measure the structural shape of real DDI-2013 sentences.

Design input for the generator: how many entities a sentence carries, how those
entities co-occur syntactically, what fraction of NONE pairs sit in sentences that
also assert an interaction, and how often sentences carry unresolved discourse
reference.

Run on TRAIN documents only. dev/val come out of the same Train directory at the
document level, so passing doc_ids is what keeps this off the eval sets.

Everything here reads unlabelled structure except the hard-negative rate and the
positives-per-sentence counts, which use gold relations. Those two are labelled
measurements: use them to understand the corpus, and if you generate against them,
report it as a design input with a guidelines-only prior as the control arm.
"""
import re
import itertools
from collections import Counter, defaultdict

from ddi.data import load_brat_docs, make_sentence_level

# entity types in DDI-2013: drug, brand, group, drug_n
GROUP_TYPES = {"group", "GROUP"} 

COORD = re.compile(r"^(,?\s*(and|or|and/or|,)\s*)+$", re.I)
CONTRAST = re.compile(r"\b(whereas|while|in contrast|unlike|conversely|however|although)\b", re.I)
ANAPHORA = re.compile(
    r"^\s*(this|these|those|it|they|such|both|either|the (drug|agent|combination|"
    r"latter|former|above)|as (noted|described|mentioned))\b", re.I)
AFOREMENTIONED = re.compile(r"\b(aforementioned|as (noted|described|mentioned) above|the latter|the former)\b", re.I)


def between(text, e1, e2):
    """Text strictly between two entity spans, ordered by position."""
    a, b = sorted([e1, e2], key=lambda e: e.locations.begin())
    return text[a.locations.end():b.locations.begin()]


def classify_pair(sent, e1, e2, rel_count):
    """Non-exclusive structural flags for how a pair co-occurs."""
    gap = between(sent.text, e1, e2)
    flags = set()
    if COORD.match(gap.strip()) or (len(gap) < 25 and COORD.search(gap.strip())):
        flags.add("coordination")
    if (e1.type in GROUP_TYPES) != (e2.type in GROUP_TYPES):
        flags.add("group_member")
    if CONTRAST.search(gap):
        flags.add("contrast")
    if rel_count[e1.id] >= 2 or rel_count[e2.id] >= 2:
        flags.add("hub")
    if not flags:
        flags.add("incidental")
    return flags


PRECEDENCE = ["hub", "group_member", "coordination", "contrast", "incidental"]


def primary(flags):
    for f in PRECEDENCE:
        if f in flags:
            return f
    return "incidental"


def analyse(docs):
    ents_per_sent = Counter()
    pairs_per_sent = Counter()
    pos_per_sent = Counter()
    entity_types = Counter()
    flag_counts = Counter()
    primary_counts = Counter()
    primary_by_label = defaultdict(Counter)
    token_gap = []
    sent_lengths = []

    n_sent = n_sent_multi = 0
    n_anaphora = n_aforementioned = 0
    n_pairs = n_pos = 0
    n_none_in_mixed = n_none_total = 0
    n_multi_pos_sent = 0
    by_register = defaultdict(Counter)

    for doc in docs:
        for sent in make_sentence_level(doc):
            n_sent += 1
            ents = sorted(sent.entities, key=lambda e: e.locations.begin())
            ents_per_sent[len(ents)] += 1
            sent_lengths.append(len(sent.text.split()))
            for e in ents:
                entity_types[e.type] += 1

            if ANAPHORA.match(sent.text):
                n_anaphora += 1
            if AFOREMENTIONED.search(sent.text):
                n_aforementioned += 1

            if len(ents) < 2:
                continue
            n_sent_multi += 1
            by_register[sent.register]["sentences"] += 1

            labelled = {}
            rel_count = Counter()
            for rel in sent.relations:
                a1, a2 = rel.arguments["Arg1"], rel.arguments["Arg2"]
                labelled[frozenset((a1, a2))] = rel.type
                rel_count[a1] += 1
                rel_count[a2] += 1

            n_pos_here = len(labelled)
            pos_per_sent[n_pos_here] += 1
            if n_pos_here >= 2:
                n_multi_pos_sent += 1

            pairs = list(itertools.combinations(ents, 2))
            pairs_per_sent[len(pairs)] += 1
            n_pairs += len(pairs)
            by_register[sent.register]["pairs"] += len(pairs)

            for e1, e2 in pairs:
                label = labelled.get(frozenset((e1.id, e2.id)), "NONE")
                flags = classify_pair(sent, e1, e2, rel_count)
                for f in flags:
                    flag_counts[f] += 1
                p = primary(flags)
                primary_counts[p] += 1
                primary_by_label[label][p] += 1
                token_gap.append(len(between(sent.text, e1, e2).split()))

                if label == "NONE":
                    n_none_total += 1
                    if n_pos_here >= 1:
                        n_none_in_mixed += 1
                else:
                    n_pos += 1
                    by_register[sent.register]["positives"] += 1

    return dict(
        n_sent=n_sent, n_sent_multi=n_sent_multi,
        ents_per_sent=ents_per_sent, pairs_per_sent=pairs_per_sent,
        pos_per_sent=pos_per_sent, entity_types=entity_types,
        flag_counts=flag_counts, primary_counts=primary_counts,
        primary_by_label=primary_by_label,
        token_gap=token_gap, sent_lengths=sent_lengths,
        n_anaphora=n_anaphora, n_aforementioned=n_aforementioned,
        n_pairs=n_pairs, n_pos=n_pos,
        n_none_in_mixed=n_none_in_mixed, n_none_total=n_none_total,
        n_multi_pos_sent=n_multi_pos_sent, by_register=by_register,
    )


def _pct(part, whole):
    return f"{part}/{whole} = {part / whole:.3f}" if whole else "n/a"


def report(r):
    print("=" * 62)
    print("SENTENCES")
    print(f"  total {r['n_sent']}, with >=2 entities {_pct(r['n_sent_multi'], r['n_sent'])}")
    lens = sorted(r["sent_lengths"])
    print(f"  length words: median {lens[len(lens)//2]}, p90 {lens[int(len(lens)*0.9)]}")

    print("\nENTITIES PER SENTENCE  (this is the number to generate against)")
    tot = sum(r["ents_per_sent"].values())
    for k in sorted(r["ents_per_sent"]):
        if k <= 12:
            print(f"  {k:>2}: {r['ents_per_sent'][k]:>5}  {r['ents_per_sent'][k]/tot:.3f}")
    multi = {k: v for k, v in r["ents_per_sent"].items() if k >= 2}
    mtot = sum(multi.values())
    print("  conditional on >=2:")
    for k in sorted(multi):
        if k <= 12:
            print(f"  {k:>2}: {multi[k]/mtot:.3f}")

    print("\nENTITY TYPES  (informs p_group in the vocab sampler)")
    ttot = sum(r["entity_types"].values())
    for t, c in r["entity_types"].most_common():
        print(f"  {t:<10} {c:>6}  {c/ttot:.3f}")

    print("\nPAIRS")
    print(f"  total {r['n_pairs']}, positive {_pct(r['n_pos'], r['n_pairs'])}")
    print(f"  NONE rate {1 - r['n_pos']/r['n_pairs']:.3f}")

    print("\nHARD NEGATIVE RATE  (the headline number for negative design)")
    print(f"  NONE pairs in sentences that also assert an interaction: "
          f"{_pct(r['n_none_in_mixed'], r['n_none_total'])}")
    print(f"  sentences with >=2 asserted interactions: "
          f"{_pct(r['n_multi_pos_sent'], r['n_sent_multi'])}")

    print("\nPOSITIVES PER MULTI-ENTITY SENTENCE")
    ptot = sum(r["pos_per_sent"].values())
    for k in sorted(r["pos_per_sent"]):
        if k <= 8:
            print(f"  {k:>2}: {r['pos_per_sent'][k]/ptot:.3f}")

    print("\nCO-OCCURRENCE CONSTRUCTION  (non-exclusive flags, share of all pairs)")
    for f, c in r["flag_counts"].most_common():
        print(f"  {f:<14} {c/r['n_pairs']:.3f}")
    print("  primary (precedence " + " > ".join(PRECEDENCE) + "):")
    for f, c in r["primary_counts"].most_common():
        print(f"  {f:<14} {c/r['n_pairs']:.3f}")

    print("\n  primary construction, split by label:")
    for label in ["NONE", "MECHANISM", "EFFECT", "ADVISE", "INT"]:
        row = r["primary_by_label"].get(label)
        if not row:
            continue
        n = sum(row.values())
        cells = "  ".join(f"{f} {row[f]/n:.2f}" for f in PRECEDENCE if row[f])
        print(f"  {label:<10} (n={n:>5})  {cells}")

    print("\nTOKEN GAP BETWEEN PAIR MEMBERS")
    g = sorted(r["token_gap"])
    print(f"  median {g[len(g)//2]}, p90 {g[int(len(g)*0.9)]}, max {g[-1]}")

    print("\nDISCOURSE REFERENCE  (what a standalone generated sentence lacks)")
    print(f"  sentence-initial anaphora: {_pct(r['n_anaphora'], r['n_sent'])}")
    print(f"  aforementioned/latter/former: {_pct(r['n_aforementioned'], r['n_sent'])}")

    print("\nBY REGISTER")
    for reg, c in r["by_register"].items():
        pairs_per = c["pairs"] / c["sentences"] if c["sentences"] else 0
        pos_rate = c["positives"] / c["pairs"] if c["pairs"] else 0
        print(f"  {reg:<10} multi-entity sents {c['sentences']:>5}, "
              f"pairs/sent {pairs_per:.2f}, positive rate {pos_rate:.3f}")
    print("=" * 62)


def sample_multi_entity(docs, min_ents=3, n=15, require_positive=True, seed=0):
    """Print real sentences to read. The counts tell you the distribution;
    reading the sentences tells you what construction to actually generate."""
    import random
    hits = []
    for doc in docs:
        for sent in make_sentence_level(doc):
            if len(sent.entities) < min_ents:
                continue
            if require_positive and not sent.relations:
                continue
            hits.append(sent)
    random.Random(seed).shuffle(hits)
    for sent in hits[:n]:
        ids = {e.id: e for e in sent.entities}
        rels = ", ".join(
            f"{ids[r.arguments['Arg1']].text}~{ids[r.arguments['Arg2']].text}:{r.type}"
            for r in sent.relations if r.arguments['Arg1'] in ids and r.arguments['Arg2'] in ids)
        print(f"\n[{sent.register}] {sent.text}")
        print(f"   entities: {[(e.text, e.type) for e in sent.entities]}")
        print(f"   relations: {rels or 'none'}")
    print(f"\n({len(hits)} sentences matched)")


if __name__ == "__main__":
    docs = load_brat_docs("Train")
    # TODO: restrict to the train split's doc_ids before using any of this as a
    # design input, or you are fitting the generator to dev/val structure.
    # docs = [d for d in docs if d.doc_id in TRAIN_DOC_IDS]
    report(analyse(docs))