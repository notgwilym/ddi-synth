import os
from bioc import brat
import spacy

CORPUS = "DDICorpusBrat"


def load_brat_docs(split="Train"):
    docs = []
    for directory in ["DrugBank", "MedLine"]:
        dir_path = f"{CORPUS}/{split}/{directory}/"
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                ann_filename = filename.replace(".txt", ".ann")
                with open(os.path.join(dir_path, ann_filename)) as ann_fp, open(
                    os.path.join(dir_path, filename)
                ) as text_fp:
                    doc = brat.load(text_fp, ann_fp)
                    doc.register = directory
                    doc.doc_id = filename.replace(".txt", "")
                    docs.append(doc)
    return docs


from bioc.brat.datastructure import BratDocument


def make_sentence_level(doc, nlp=spacy.load("en_core_web_sm")):
    parsed = nlp(doc.text)
    sent_docs = []
    for i, sent in enumerate(parsed.sents):
        sent_entities = [
            e
            for e in doc.entities
            if sent.start_char <= e.locations.begin()
            and e.locations.end() <= sent.end_char
        ]
        sent_entities = [e.shift(-sent.start_char) for e in sent_entities]
        sent_entity_ids = {e.id for e in sent_entities}
        sent_relations = [
            rel
            for rel in doc.relations
            if all(e_id in sent_entity_ids for e_id in rel.arguments.values())
        ]
        sent_doc = BratDocument()
        sent_doc.text = sent.text
        sent_doc.annotations += sent_entities
        sent_doc.annotations += sent_relations
        sent_doc.register = doc.register
        sent_doc.sent_id = f"human:{doc.register}:{doc.doc_id}:s{i}"
        sent_docs.append(sent_doc)
    return sent_docs


import itertools

MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
POSITIVE_LABELS = ["ADVISE", "EFFECT", "INT", "MECHANISM"]
ALL_LABELS = ["NONE"] + POSITIVE_LABELS


RENDER_MODES = ("markers", "mask_targets", "mask_all")


class Overlap(Exception):
    """Spans to be rendered overlap; the instance can't be rendered safely."""


def render_pair(text, e1, e2, others=(), mode="markers"):
    b1, n1 = e1.locations.begin(), e1.locations.end()
    b2, n2 = e2.locations.begin(), e2.locations.end()

    if mode == "markers":
        # insertion, not replacement: tolerates nested spans, byte-identical to the
        # original make_pair_instances
        inserts = sorted([(b1, MARKERS[0]), (n1, MARKERS[1]),
                          (b2, MARKERS[2]), (n2, MARKERS[3])],
                         key=lambda x: x[0], reverse=True)
        out = text
        for pos, tag in inserts:
            out = out[:pos] + tag + out[pos:]
        return out

    if mode not in ("mask_targets", "mask_all"):
        raise ValueError(f"unknown render mode {mode!r}")

    edits = [(b1, n1, "drug1"), (b2, n2, "drug2")]
    if mode == "mask_all":
        for e in others:
            b, n = e.locations.begin(), e.locations.end()
            if (b, n) in ((b1, n1), (b2, n2)):
                continue
            if not (n <= b1 or b >= n1) or not (n <= b2 or b >= n2):
                continue
            edits.append((b, n, "drug0"))

    edits.sort(key=lambda x: x[0])
    for (_, a_end, _), (c_begin, _, _) in zip(edits, edits[1:]):
        if c_begin < a_end:
            raise Overlap(f"overlapping spans: {[(b, n) for b, n, _ in edits]}")

    out = text
    for b, n, repl in reversed(edits):
        out = out[:b] + repl + out[n:]
    return out


def make_pair_instances(doc, mode="markers"):
    candidate_to_label = {}
    for rel in doc.relations:
        candidate_to_label[(rel.arguments["Arg1"], rel.arguments["Arg2"])] = rel.type

    entities = sorted(doc.entities, key=lambda e: e.locations.begin())
    labelled_data, n_overlap = [], 0
    for e1, e2 in itertools.combinations(entities, 2):
        try:
            new_text = render_pair(doc.text, e1, e2, entities, mode)
        except Overlap:
            n_overlap += 1
            continue
        label = (candidate_to_label.get((e1.id, e2.id))
                 or candidate_to_label.get((e2.id, e1.id))
                 or "NONE")
        labelled_data.append({"text": new_text, "label": label,
                              "source": doc.register, "sent_id": doc.sent_id})
    if n_overlap:
        print(f"warning: {n_overlap} pairs skipped for overlapping spans in {doc.sent_id}")
    return labelled_data


from sklearn.model_selection import train_test_split


def build_human(dev_split=0.15, val_split=0.15, seed=42):
    docs = load_brat_docs(split="Train")
    
    train_docs, holdout_docs = train_test_split(
        docs, test_size=dev_split + val_split, random_state=seed
    )

    rel_val = val_split / (dev_split + val_split)
    dev_docs, val_docs = train_test_split(
        holdout_docs, test_size=rel_val, random_state=seed
    )

    nlp = spacy.load("en_core_web_sm")

    def to_instances(doc_list):
        out = []
        for doc in doc_list:
            for sent in make_sentence_level(doc, nlp):
                out.extend(make_pair_instances(sent))
        return out

    return to_instances(train_docs), to_instances(dev_docs), to_instances(val_docs)


import random


def downsample_train_negatives(records, negative_ratio=None, seed=42):
    rng = random.Random(seed)
    positive_data = [x for x in records if x["label"] != "NONE"]
    negative_data = [x for x in records if x["label"] == "NONE"]

    if negative_ratio is not None:
        k = min(len(negative_data), int(negative_ratio * len(positive_data)))
        smaller_data = positive_data + rng.sample(negative_data, k)
    else:
        smaller_data = positive_data + negative_data

    rng.shuffle(smaller_data)
    return smaller_data