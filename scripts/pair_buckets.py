"""Bucket eval performance by the entity count of the source sentence.

Motivation: 53% of real training pairs come from ~32 sentences carrying 20+ entities
(DrugBank enumerations). Synthetic generation produces ~1 pair per sentence, so it has
no examples of the "two drugs far apart in a long list do not interact" construction.
If that is a real gap, synthetic precision should collapse on high-entity buckets while
holding on 2-3 entity ones. Structural, so it survives the masking result.

Also dumps the monster sentences, because their content decides whether this is a
genuine corpus construction or the TRACRIUM-style duplicate-block artefact.
"""
import itertools
from collections import Counter, defaultdict

from ddi.data import load_brat_docs, make_sentence_level, ALL_LABELS, POSITIVE_LABELS

BUCKETS = [(2, 2), (3, 3), (4, 5), (6, 9), (10, 19), (20, 10 ** 6)]


def bucket_of(n):
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
    return "other"


def sent_entity_counts(docs):
    """sent_id -> number of entities. Keys match the sent_id on instances."""
    counts = {}
    for doc in docs:
        for sent in make_sentence_level(doc):
            counts[sent.sent_id] = len(sent.entities)
    return counts


def dump_big_sentences(docs, min_ents=20, max_show=40, chars=400):
    """Print the sentences that carry most of the corpus. Read these before
    designing anything: they decide whether the mass is a real enumeration
    construction or a sentence-splitting / duplicate-block artefact."""
    hits = []
    for doc in docs:
        for sent in make_sentence_level(doc):
            if len(sent.entities) >= min_ents:
                hits.append((len(sent.entities), doc.register, doc.doc_id, sent))
    hits.sort(reverse=True, key=lambda h: h[0])

    total_pairs = sum(n * (n - 1) // 2 for n, _, _, _ in hits)
    print(f"{len(hits)} sentences with >={min_ents} entities, "
          f"contributing {total_pairs} pairs\n")

    for n, register, doc_id, sent in hits[:max_show]:
        n_rel = len(sent.relations)
        pairs = n * (n - 1) // 2
        print(f"[{register}:{doc_id}] {n} entities, {pairs} pairs, {n_rel} relations")
        print(f"  {sent.text[:chars]}{'...' if len(sent.text) > chars else ''}")
        print(f"  types: {dict(Counter(e.type for e in sent.entities))}")
        # a duplicate-block artefact shows up as repeated entity surface forms
        surf = Counter(e.text.lower() for e in sent.entities)
        dupes = {k: v for k, v in surf.items() if v > 1}
        if dupes:
            print(f"  REPEATED SURFACE FORMS (possible duplicate block): {dupes}")
        print()


def pair_share_by_bucket(docs):
    """Where the training mass actually is."""
    pairs = Counter()
    sents = Counter()
    for doc in docs:
        for sent in make_sentence_level(doc):
            n = len(sent.entities)
            if n < 2:
                continue
            b = bucket_of(n)
            sents[b] += 1
            pairs[b] += n * (n - 1) // 2
    tot_p, tot_s = sum(pairs.values()), sum(sents.values())
    print(f"{'bucket':<8} {'sents':>6} {'sent%':>7} {'pairs':>7} {'pair%':>7}")
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
        print(f"{b:<8} {sents[b]:>6} {sents[b]/tot_s:>7.3f} "
              f"{pairs[b]:>7} {pairs[b]/tot_p:>7.3f}")
    print(f"{'total':<8} {tot_s:>6} {'':>7} {tot_p:>7}")


def bucket_scores(val_records, y_pred, counts):
    """Micro P/R/F1 over positive classes, per entity-count bucket.

    val_records : the eval instances (need 'label' and 'sent_id')
    y_pred      : list of predicted label STRINGS, same order
    counts      : sent_id -> n_entities, from sent_entity_counts
    """
    assert len(val_records) == len(y_pred), "predictions misaligned with records"
    pos = set(POSITIVE_LABELS)
    agg = defaultdict(lambda: Counter())
    missing = 0

    for r, pred in zip(val_records, y_pred):
        n = counts.get(r["sent_id"])
        if n is None:
            missing += 1
            continue
        b = bucket_of(n)
        gold = r["label"]
        c = agg[b]
        c["n"] += 1
        if gold in pos:
            c["support"] += 1
        if pred in pos:
            c["predicted"] += 1
        if gold in pos and pred == gold:
            c["correct"] += 1

    if missing:
        print(f"warning: {missing} instances had no sent_id in the count map")

    rows = []
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
        c = agg.get(b)
        if not c or not c["n"]:
            continue
        p = c["correct"] / c["predicted"] if c["predicted"] else 0.0
        rec = c["correct"] / c["support"] if c["support"] else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        rows.append({"bucket": b, "n_pairs": c["n"], "support": c["support"],
                     "predicted": c["predicted"], "P": p, "R": rec, "F1": f1})
    return rows


def print_buckets(rows, title=""):
    print(f"\n{title}")
    print(f"{'bucket':<8} {'pairs':>7} {'gold+':>6} {'pred+':>6} "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    for r in rows:
        print(f"{r['bucket']:<8} {r['n_pairs']:>7} {r['support']:>6} "
              f"{r['predicted']:>6} {r['P']:>6.3f} {r['R']:>6.3f} {r['F1']:>6.3f}")


def compare(rows_a, rows_b, name_a="human", name_b="synthetic"):
    """Side by side. The signal to look for: does the P gap widen with bucket size?"""
    by_b = {r["bucket"]: r for r in rows_b}
    print(f"\n{'bucket':<8} {'pairs':>7} "
          f"{name_a[:6]+' P':>9} {name_b[:6]+' P':>9} {'P gap':>7} "
          f"{name_a[:6]+' R':>9} {name_b[:6]+' R':>9}")
    for a in rows_a:
        b = by_b.get(a["bucket"])
        if not b:
            continue
        print(f"{a['bucket']:<8} {a['n_pairs']:>7} "
              f"{a['P']:>9.3f} {b['P']:>9.3f} {a['P']-b['P']:>7.3f} "
              f"{a['R']:>9.3f} {b['R']:>9.3f}")