import itertools
from ddi.data import build_human, downsample_train_negatives
from ddi.train import train_and_eval
from ddi.experiment import log_run

train, val = build_human()

BASE = {"model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "epochs": 3, "lr": 2e-5, "batch_size": 32, "max_length": 256}

for neg_ratio in [2, 5, None]:
    tr = downsample_train_negatives(train, neg_ratio, seed=42)
    for seed in [0, 1, 2]:
        cfg = {**BASE, "neg_ratio": neg_ratio, "seed": seed, "dataset": "human"}
        m = train_and_eval(cfg, tr, val)
        log_run(cfg, m)
        print(f"neg={str(neg_ratio):4} seed={seed}  f1={m['micro_f1_pos']:.3f}")