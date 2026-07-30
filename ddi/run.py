"""Train on a manifested dataset, evaluate on a pinned eval set, log with provenance."""
from .manifest import load_dataset
from .experiment import log_run
from .train import train_and_eval


def run_training(train_id, eval_id, cfg, notes="", eval_instances=None,
                 allow_mode_mismatch=False):
    train_instances, train_man = load_dataset(train_id)
    if eval_instances is None:
        eval_instances, eval_man = load_dataset(eval_id)
        if eval_man["provenance"].startswith("synthetic"):
            raise ValueError("refusing to evaluate on a synthetic set")
        train_mode = train_man.get("render_mode", "markers")
        eval_mode = eval_man.get("render_mode", "markers")
        cfg_mode = cfg.get("render_mode", "markers")
        if train_mode != eval_mode and not allow_mode_mismatch:
            raise ValueError(f"render mode mismatch: train={train_mode} eval={eval_mode}")
        if cfg_mode != train_mode:
            raise ValueError(f"cfg render_mode={cfg_mode} but data is {train_mode}")

    metrics = train_and_eval(cfg, train_instances, eval_instances)
    run_id = log_run(cfg, metrics, notes=notes, train_id=train_id, eval_id=eval_id)
    return run_id, metrics