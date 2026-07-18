import os
from bioc import brat
import spacy

CORPUS = "DDICorpusBrat"

def load_brat_docs(split = "Train"):
  docs = []
  for directory in ['DrugBank', 'MedLine']:
    dir_path = f'{CORPUS}/{split}/{directory}/'
    for filename in os.listdir(dir_path):
      if filename.endswith('.txt'):
        ann_filename = filename.replace('.txt', '.ann')
        with open(os.path.join(dir_path, ann_filename)) as ann_fp, open(os.path.join(dir_path, filename)) as text_fp:
          doc = brat.load(text_fp, ann_fp)
          doc.register = directory
          docs.append(doc)
  return docs

from bioc.brat.datastructure import BratDocument

def make_sentence_level(doc, nlp = spacy.load("en_core_web_sm")):
  parsed = nlp(doc.text)
  sent_docs = []
  for sent in parsed.sents:
    sent_entities = [ e for e in doc.entities if sent.start_char <= e.locations.begin() and e.locations.end() <= sent.end_char ]
    sent_entities = [ e.shift(-sent.start_char) for e in sent_entities ]
    sent_entity_ids = { e.id for e in sent_entities }
    sent_relations = [ rel for rel in doc.relations if all(e_id in sent_entity_ids for e_id in rel.arguments.values()) ]
    sent_doc = BratDocument() 
    sent_doc.text = sent.text
    sent_doc.annotations += sent_entities
    sent_doc.annotations += sent_relations
    sent_doc.register = doc.register
    sent_docs.append(sent_doc)
  return sent_docs

import itertools

def make_pair_instances(doc):
    candidate_to_label = {}
    for rel in doc.relations:
        candidate_to_label[(rel.arguments['Arg1'], rel.arguments['Arg2'])] = rel.type

    entities = sorted(doc.entities, key=lambda e: e.locations.begin())

    labelled_data = []
    for e1, e2 in itertools.combinations(entities, 2):
        inserts = [(e1.locations.begin(), '[E1]'), (e1.locations.end(), '[/E1]'),
                   (e2.locations.begin(), '[E2]'), (e2.locations.end(), '[/E2]')]
        inserts = sorted(inserts, key=lambda x: x[0], reverse=True)
        new_text = doc.text
        for pos, tag in inserts:
            new_text = new_text[:pos] + tag + new_text[pos:]
        label = candidate_to_label.get((e1.id, e2.id)) or candidate_to_label.get((e2.id, e1.id)) or 'NONE'
        labelled_data.append({'text': new_text, 'label': label, 'source': doc.register})
    return labelled_data