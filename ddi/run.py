"""Train on a manifested dataset, evaluate on a pinned eval set, log with provenance."""
from .manifest import load_dataset
from .experiment import log_run
from .train import train_and_eval


def run_training(train_id, eval_id, cfg, notes="", eval_instances=None):
    train_instances, train_man = load_dataset(train_id)
    if eval_instances is None:
        eval_instances, eval_man = load_dataset(eval_id)
        if eval_man["provenance"].startswith("synthetic"):
            raise ValueError("refusing to evaluate on a synthetic set")

    metrics = train_and_eval(cfg, train_instances, eval_instances)
    run_id = log_run(cfg, metrics, notes=notes, train_id=train_id, eval_id=eval_id)
    return run_id, metrics