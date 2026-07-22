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


def make_pair_instances(doc):
    candidate_to_label = {}
    for rel in doc.relations:
        candidate_to_label[(rel.arguments["Arg1"], rel.arguments["Arg2"])] = rel.type

    entities = sorted(doc.entities, key=lambda e: e.locations.begin())

    labelled_data = []
    for e1, e2 in itertools.combinations(entities, 2):
        inserts = [
            (e1.locations.begin(), MARKERS[0]),
            (e1.locations.end(), MARKERS[1]),
            (e2.locations.begin(), MARKERS[2]),
            (e2.locations.end(), MARKERS[3]),
        ]
        inserts = sorted(inserts, key=lambda x: x[0], reverse=True)
        new_text = doc.text
        for pos, tag in inserts:
            new_text = new_text[:pos] + tag + new_text[pos:]
        label = (
            candidate_to_label.get((e1.id, e2.id))
            or candidate_to_label.get((e2.id, e1.id)) # normally redundant
            or "NONE"
        )
        labelled_data.append({"text": new_text, "label": label, "source": doc.register, "sent_id": doc.sent_id})
    return labelled_data


from sklearn.model_selection import train_test_split


def build_human(val_split=0.2, seed=42):
    docs = load_brat_docs(split="Train")
    train_docs, val_docs = train_test_split(
        docs, test_size=val_split, random_state=seed
    )

    nlp = spacy.load("en_core_web_sm")
    train_instances = []
    for doc in train_docs:
        for sent in make_sentence_level(doc, nlp):
            train_instances.extend(make_pair_instances(sent))
    val_instances = []
    for doc in val_docs:
        for sent in make_sentence_level(doc, nlp):
            val_instances.extend(make_pair_instances(sent))
    return (train_instances, val_instances)


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
