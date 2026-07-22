import time 
import numpy as np
import torch

from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from sklearn.metrics import precision_recall_fscore_support
from .data import ALL_LABELS, MARKERS, POSITIVE_LABELS

def build_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.add_tokens(MARKERS)
    return tok

def score(y_true, y_pred, sources, label2id):
    pos_ids = [label2id[l] for l in POSITIVE_LABELS]
    m = {}
    # headline - micro f1 over positive labels
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=pos_ids, 
                                                 average="micro", zero_division=0)
    m["micro_f1_pos"], m["micro_p_pos"], m["micro_r_pos"] = float(f), float(p), float(r)
    # macro f1 over positive labels
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=pos_ids, 
                                                 average="macro", zero_division=0)
    m["macro_f1_pos"], m["macro_p_pos"], m["macro_r_pos"] = float(f), float(p), float(r)
    # per-label metrics
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=pos_ids, 
                                                 average=None, zero_division=0)
    for i, l in enumerate(POSITIVE_LABELS):
        m[f"f1_{l}"], m[f"p_{l}"], m[f"r_{l}"], m[f"support_{l}"] = float(f[i]), float(p[i]), float(r[i]), float(s[i])
    # micro f1 masked by source
    for source in sorted(set(sources)):
        keep = [s == source for s in sources]
        p, r, f, _ = precision_recall_fscore_support(
            [y for y, k in zip(y_true, keep) if k],
            [y for y, k in zip(y_pred, keep) if k],
            labels=pos_ids,
            average="micro",
            zero_division=0,
        )
        m[f"micro_f1_pos_{source}"], m[f"micro_p_pos_{source}"], m[f"micro_r_pos_{source}"] = float(f), float(p), float(r)
    return m

def train_and_eval(cfg, train_records, val_records):
    def prep(recs):
        ds = Dataset.from_list([
            {"text": r["text"], "labels": label2id[r["label"]]} for r in recs
        ])
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=cfg["max_length"]),
            batched=True,
            remove_columns=["text"],
        )
    
    t0 = time.time()
    set_seed(cfg["seed"])
    
    label2id = {l: i for i, l in enumerate(ALL_LABELS)}
    id2label = {i: l for l, i in label2id.items()}
    
    tok = build_tokenizer(cfg)
    train_ds = prep(train_records)
    val_ds = prep(val_records)

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=len(ALL_LABELS), id2label=id2label, label2id=label2id, use_safetensors=True)
    model.resize_token_embeddings(len(tok))

    training_args = TrainingArguments(
        output_dir="/tmp/ddi_out",
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=128,
        num_train_epochs=cfg["epochs"],
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=False,
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=cfg["seed"],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()
    
    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids
    sources = [r["source"] for r in val_records]
    
    m = score(y_true, y_pred, sources, label2id)
    m["train_size"] = len(train_records)
    m["val_size"] = len(val_records)
    m["train_time"] = round(time.time() - t0, 1)
    return m