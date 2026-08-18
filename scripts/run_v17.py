"""v17 end to end, unattended: generate, resolve, gate, train, report.

Run under tmux. Writes a markdown report to reports/v17-<timestamp>.md and prints the
same to stdout, so you can read the outcome from a phone with `tail`.

    tmux new -s v17
    cd /mnt/primary/synth_data_creation
    mkdir -p logs reports
    python scripts/run_v17.py 2>&1 | tee logs/v17.log
    # ctrl-b then d

What it is designed to survive:
  worker down at start          polls until it answers, then begins
  worker dies mid-generation    generate_raw appends and resumes by spec index; the
                                script retries the whole generate call up to 5 times
  pod dies                      raw jsonl is on NFS; rerun the same command and it
                                picks up from where it stopped
  a frame producing garbage     nothing is dropped automatically, but the report says
                                which frames broke their constraints

What it will NOT do: fix a broken frame. study_only is set to weight 0 below because 13
of 15 smoke sentences reported an outcome the spec forbade. Every other frame passed.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ddi.data import build_human
from ddi.vocab import build_vocab
from ddi import prompt as P
from ddi.prompt import make_specs, make_sample_fn, fingerprint, FRAMES
from ddi.resolve import v14_sample_to_instances, generation_records
from ddi.synth import generate_raw, build_dataset_from_raw, RAW
from ddi.manifest import load_dataset
from ddi.train import train_and_eval
from ddi.experiment import log_run
from ddi import gates, divergence as dv

V14_ID = "20260807-123340-ff79db"
V15_ID = "20260816-005908-3d6539"

# Reference numbers, same dev set, 3 seeds each. Printed in the report so the new number
# is readable without looking anything up.
REFERENCE = [
    ("human, filtered pool", 0.790, 0.754, 0.829),
    ("human, full pool", 0.800, 0.768, 0.835),
    ("v14", 0.379, 0.302, 0.511),
    ("v14 judged", 0.387, 0.307, 0.525),
    ("v14 judged, verifier-pruned", 0.408, 0.304, 0.623),
    ("v14 judged, random-pruned", 0.367, 0.292, 0.495),
    ("v14 low-role", 0.344, 0.288, 0.428),
    ("v15", 0.333, 0.272, 0.430),
    ("v13", 0.280, 0.182, 0.610),
]

MARK = re.compile(r"\[/?E[12]\]")
PK_WORDS = re.compile(
    r"\b(concentration|level|AUC|Cmax|half-?life|clearance|absorb\w+|absorption|"
    r"metaboli\w+|bioavailab\w+|exposure|protein binding)\b", re.I)
ASSERTS = re.compile(
    r"\b(increas\w+|decreas\w+|reduc\w+|rais\w+|lower\w+|inhibit\w+|induc\w+|"
    r"potentiat\w+|antagonis\w+|interact\w+|result\w+ in|led to|caus\w+)\b", re.I)
JOINT = re.compile(
    r"\b(combination|regimen|combined|together)\b[^.]{0,80}"
    r"\b(tolerat\w+|benefit\w*|improv\w+|effective|efficac\w+|additive|safe\w*|"
    r"outcome\w*|response)\b"
    r"|\b(suggest\w+|observ\w+|appear\w+|indicat\w+)\b[^.]{0,60}"
    r"\b(benefit\w*|improv\w+|efficac\w+|additive)\b", re.I)

MUST = {
    "denial": re.compile(r"\b(no|not|without|unchanged|unaffected|neither)\b", re.I),
    "incompatible": re.compile(r"\b(mix\w+|precipitat\w+|solution|infusion|syringe|"
                               r"cloud\w+|unstable|instability)\b", re.I),
    "enzyme_only": re.compile(r"\b(CYP\w*|P-?glycoprotein|UGT\w*|OATP\w*)\b", re.I),
    "out_of_scope": re.compile(r"\b(juice|wort|meal|tea|milk|smoking|liquorice|ginkgo)\b", re.I),
    "sequential": re.compile(r"\b(after|before|stopp\w+|discontinu\w+|withdraw\w+|"
                             r"washout|subsequent\w*|previously)\b", re.I),
    "effect_pd": re.compile(r"\b(synerg\w+|additive|potentiat\w+|antagoni\w+|enhanc\w+|"
                            r"oppos\w+|blunt\w+)\b", re.I),
    "appositive": re.compile(r"\b(including|such as)\b", re.I),
    "title": re.compile(r"^[^.]{0,60}:"),
    "contradictory": re.compile(r"\b(although|however|not been reported|nevertheless|"
                                r"even though|despite)\b", re.I),
    "effect_protect": re.compile(r"\b(protect\w+|reduc\w+|less\b|lower\b|attenuat\w+)\b", re.I),
}
# §4.5.11 priority order: an effect or advise sentence that mentions a PK change is
# relabelled mechanism by an annotator, so these are label errors, not style errors.
MUST_NOT = {
    "regimen": ASSERTS,
    "incompatible": re.compile(r"\b(plasma|serum|absorption|metaboli\w+|clearance|"
                               r"in the body|systemic)\b", re.I),
    "effect": PK_WORDS, "effect_pd": PK_WORDS, "effect_protect": PK_WORDS,
    "effect_failure": PK_WORDS, "advise": PK_WORDS, "advise_reason": PK_WORDS,
    "coordinate": PK_WORDS, "appositive": PK_WORDS,
}


def wait_for_worker(client, model, effort, every=120, limit=None):
    """The autoscaler kills jobs stuck in PENDING after ~305s and reclaims idle workers.
    Each ping is itself a scale-up request, so polling is the only thing that helps."""
    t0 = time.time()
    while True:
        try:
            client.responses.create(model=model, input="ping", max_output_tokens=16,
                                    reasoning={"effort": effort} if effort else {})
            print(f"[{time.strftime('%H:%M:%S')}] {model} up after "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
            return True
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] {type(e).__name__}: {str(e)[:70]}",
                  flush=True)
            if limit and time.time() - t0 > limit:
                return False
            time.sleep(every)


def generate_with_retry(specs, fn, gen_id, workers, client, model, effort, tries=5):
    """generate_raw resumes by spec index, so a retry costs only the in-flight batch."""
    for attempt in range(1, tries + 1):
        try:
            generate_raw(specs, fn, gen_id=gen_id, max_workers=workers)
            done = sum(1 for line in (RAW / f"{gen_id}.jsonl").read_text().splitlines()
                       if line and not json.loads(line).get("error"))
            if done >= 0.9 * len(specs):
                return done
            print(f"attempt {attempt}: only {done}/{len(specs)} ok, retrying",
                  flush=True)
        except Exception as e:
            print(f"attempt {attempt} raised {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
        wait_for_worker(client, model, effort)
    return None


def frame_report(gen_id):
    """Per-frame adherence. Crude regexes: a smoke alarm, not a measurement. The
    breaks-avoid column on effect and advise frames is the one that matters, because a
    breach there means the gold label is wrong rather than the prose being off."""
    out = defaultdict(list)
    n_err = 0
    for line in (RAW / f"{gen_id}.jsonl").read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r.get("error") or not r.get("sample"):
            n_err += 1
            continue
        out[r["spec"]["frame"]].append((r["spec"], r["sample"]["sentence"]))

    rows, examples = [], {}
    for frame in sorted(out):
        got = out[frame]
        must, mustnot = MUST.get(frame), MUST_NOT.get(frame)
        hit = sum(bool(must.search(t)) for _, t in got) if must else None
        miss = sum(bool(mustnot.search(t)) for _, t in got) if mustnot else None
        rows.append({"frame": frame, "n": len(got),
                     "has_marker": round(hit / len(got), 2) if must else None,
                     "breaks_avoid": round(miss / len(got), 2) if mustnot else None})
        examples[frame] = [t for _, t in got[:3]]
        if mustnot:
            bad = [t for _, t in got if mustnot.search(t)][:2]
            if bad:
                examples[frame + " (BREACH)"] = bad

    reg = out.get("regimen", [])
    joint = sum(bool(JOINT.search(t)) for _, t in reg) / max(len(reg), 1)
    return pd.DataFrame(rows), examples, n_err, joint


def appositive_check(gen_id, instances):
    """The cross-product is four relations from one clause and the highest-value new
    construction. A silent mismatch poisons four pairs at a time, so verify the resolved
    labels against the spec rather than trusting them."""
    want = {}
    for line in (RAW / f"{gen_id}.jsonl").read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r.get("error") or r["spec"]["frame"] != "appositive":
            continue
        sid = f"synth:{gen_id}:{r['spec_index']}"
        want[sid] = len(r["spec"]["positives"])
    got = Counter()
    for r in instances:
        if r["sent_id"] in want and r["label"] != "NONE":
            got[r["sent_id"]] += 1
    if not want:
        return "no appositive specs"
    match = sum(1 for s, n in want.items() if got.get(s, 0) == n)
    return (f"{match}/{len(want)} appositive sentences resolved the full cross-product "
            f"({match / len(want):.2f}); expected 4 positives each, "
            f"median got {statistics.median(list(got.values()) or [0]):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--gen-id", default="v17-full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--api", default="responses", choices=["responses", "chat"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-generate", action="store_true",
                    help="stage 2 onward from an existing raw file")
    args = ap.parse_args()

    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    Path("reports").mkdir(exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    # ---- 0. frames off ----------------------------------------------------
    # 13 of 15 smoke sentences reported an outcome the spec forbade: the model treats
    # "no result" as itself a result. Weight redistributed to the two negative frames
    # that scored 1.00 and 0.00.
    if "study_only" in FRAMES:
        w = FRAMES["study_only"]["w"]
        FRAMES["study_only"]["w"] = 0.0
        FRAMES["regimen"]["w"] += w * 0.6
        FRAMES["denial"]["w"] += w * 0.4
        say(f"study_only disabled, {w:.3f} weight moved to regimen and denial")

    say(f"# v17 run {stamp}")
    say()
    say(f"prompt sha `{fingerprint()}`, {args.n} specs, seed {args.seed}, "
        f"{args.model} at {args.effort} effort")
    say()

    # ---- 1. data ----------------------------------------------------------
    vocab = build_vocab()
    train, dev, val = build_human()
    per_sent = Counter(r["sent_id"] for r in train)
    train_f = [r for r in train if per_sent[r["sent_id"]] < 190]

    specs = make_specs(args.n, vocab=vocab, seed=args.seed)
    fc = Counter(s["frame"] for s in specs)
    say("## frames sampled")
    say()
    say("| frame | n | share |")
    say("|---|---|---|")
    for f, c in fc.most_common():
        say(f"| {f} | {c} | {c / len(specs):.3f} |")
    say()

    # ---- 2. generate ------------------------------------------------------
    if not args.skip_generate:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ.get("IDA_LLM_BASE_URL",
                                    "http://api.llm.apps.os.dcs.gla.ac.uk/v1"),
            api_key=os.environ["IDA_LLM_API_KEY"], max_retries=5, timeout=90.0)
        if not wait_for_worker(client, args.model, args.effort):
            say("**worker never came up**")
            Path(f"reports/v17-{stamp}.md").write_text("\n".join(lines))
            sys.exit(1)

        fn = make_sample_fn(client, model=args.model, reasoning_effort=args.effort,
                            api=args.api)
        t0 = time.time()
        done = generate_with_retry(specs, fn, args.gen_id, args.workers,
                                   client, args.model, args.effort)
        if done is None:
            say("**generation failed after 5 attempts**")
            Path(f"reports/v17-{stamp}.md").write_text("\n".join(lines))
            sys.exit(1)
        say(f"generated {done}/{len(specs)} in {(time.time() - t0) / 60:.1f} min")
        say()

    # ---- 3. resolve -------------------------------------------------------
    ds_id, stats = build_dataset_from_raw(
        args.gen_id, resolver=v14_sample_to_instances, mode="markers",
        generator={"prompt_sha": fingerprint(), "model": args.model,
                   "reasoning_effort": args.effort, "version": "v17",
                   "frames_disabled": ["study_only"],
                   "note": "frames; negatives as constructions, slot-based says"},
        vocab_source=vocab.fingerprint(), seed=args.seed,
        notes=f"v17, {args.n} specs")
    inst, _ = load_dataset(ds_id)
    say(f"dataset `{ds_id}`, {len(inst)} instances from "
        f"{len({r['sent_id'] for r in inst})} sentences")
    say(f"rejects: {stats['reject_reasons']}")
    say()

    # ---- 4. frame adherence ----------------------------------------------
    fr, examples, n_err, joint = frame_report(args.gen_id)
    say("## frame adherence")
    say()
    say("`has_marker` is whether the construction appeared. `breaks_avoid` on an effect "
        "or advise frame means a PK change was mentioned, which by §4.5.11 priority "
        "order makes the gold label wrong rather than the prose merely off.")
    say()
    say(fr.to_markdown(index=False))
    say()
    say(f"joint-outcome claims in `regimen`: {joint:.3f} (v14 was 0.334 across all "
        f"zero-assertion sentences; those were what verifier pruning removed for "
        f"+0.041 F1)")
    say()
    say(appositive_check(args.gen_id, inst))
    say()

    breaches = fr[(fr.breaks_avoid.notna()) & (fr.breaks_avoid > 0.15)]
    if len(breaches):
        say("**frames breaching their avoid line above 0.15:**")
        say()
        say(breaches.to_markdown(index=False))
        say()

    # ---- 5. gates and divergence -----------------------------------------
    say("## gates")
    say()
    say("```")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gates.report(inst, records=generation_records(args.gen_id), strict=False)
    say(buf.getvalue().rstrip())
    say("```")
    say()

    say("## divergence from the corpus")
    say()
    say("Size-matched. Note that v15 was closer to the corpus on almost every "
        "measurement here and trained 0.046 worse, so this is a sanity check, not a "
        "target.")
    say()
    say(dv.compare(train_f, inst, match_size=True).to_markdown(index=False))
    say()

    # ---- 6. train ---------------------------------------------------------
    BASE = {"model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            "epochs": 3, "lr": 2e-5, "batch_size": 32, "max_length": 256,
            "neg_ratio": None, "render_mode": "markers", "synth_id": ds_id}
    rows, preds = [], None
    for seed in range(args.seeds):
        cfg = {**BASE, "seed": seed, "dataset": "v17"}
        m, p = train_and_eval(cfg, inst, dev, return_preds=True)
        log_run(cfg, m, notes=f"v17 {args.gen_id}")
        if seed == 0:
            preds = p
        rows.append({"seed": seed, "f1": m["micro_f1_pos"], "p": m["micro_p_pos"],
                     "r": m["micro_r_pos"],
                     "DrugBank": m.get("micro_f1_pos_DrugBank"),
                     "MedLine": m.get("micro_f1_pos_MedLine")})
        print(f"  v17 seed={seed} f1={m['micro_f1_pos']:.3f} "
              f"p={m['micro_p_pos']:.3f} r={m['micro_r_pos']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    f1, p_, r_ = df.f1.mean(), df.p.mean(), df.r.mean()
    sd = df.f1.std()

    say("## result")
    say()
    say(f"### v17: F1 {f1:.3f} (sd {sd:.3f}), P {p_:.3f}, R {r_:.3f}")
    say()
    say("| arm | F1 | P | R |")
    say("|---|---|---|---|")
    say(f"| **v17** | **{f1:.3f}** | **{p_:.3f}** | **{r_:.3f}** |")
    for name, a, b, c in REFERENCE:
        say(f"| {name} | {a:.3f} | {b:.3f} | {c:.3f} |")
    say()
    say(f"per seed: {', '.join(f'{x:.3f}' for x in df.f1)}")
    say(f"DrugBank {df.DrugBank.mean():.3f}, MedLine {df.MedLine.mean():.3f}")
    say()

    from sklearn.metrics import classification_report
    say("### per class")
    say()
    say("```")
    say(classification_report([r["label"] for r in dev], preds,
                              labels=["MECHANISM", "EFFECT", "ADVISE", "INT"],
                              zero_division=0))
    say("```")
    say("v14 per-class precision was MECHANISM 0.23, EFFECT 0.42, ADVISE 0.30, "
        "INT 0.22. MECHANISM is the class v17's `mechanism_mixed` frame targets.")
    say()

    # ---- 7. reading -------------------------------------------------------
    say("## how to read this")
    say()
    if f1 > 0.408 + 2 * sd:
        say(f"**{f1:.3f} beats the best previous arm** (v14 verifier-pruned, 0.408). "
            f"The frame redesign is worth keeping. Next: run the binary verifier on "
            f"v17 and prune, since that gained 0.041 on v14 and is orthogonal to this.")
    elif f1 > 0.379 + 2 * sd:
        say(f"**{f1:.3f} beats plain v14 (0.379)** but not the pruned arm (0.408). "
            f"The frames helped; pruning is still the larger effect and the two should "
            f"compose.")
    elif abs(f1 - 0.379) <= 2 * sd:
        say(f"**{f1:.3f} is indistinguishable from v14 (0.379).** Twenty frames, seven "
            f"new negative constructions and the guideline-grounded scope rules bought "
            f"nothing measurable. That is a real finding for the write-up: construction "
            f"coverage is not the binding constraint either.")
    else:
        say(f"**{f1:.3f} is below v14 (0.379).** Check the frame adherence table first: "
            f"a frame breaching its avoid line is producing mislabelled positives, and "
            f"the fix is to disable it and rerun stage 2, which is free.")
    say()
    say("The composition-matched arms on v15 all landed within 0.02 of each other, so "
        "if this number disappoints, composition is not the explanation and does not "
        "need testing again.")
    say()

    say("## examples by frame")
    say()
    for frame in sorted(examples):
        say(f"**{frame}**")
        say()
        for t in examples[frame]:
            say(f"- {t}")
        say()

    say(f"total wall clock {(time.time() - t_start) / 60:.1f} min")

    out = Path(f"reports/v17-{stamp}.md")
    out.write_text("\n".join(lines))
    print(f"\nreport written to {out}", flush=True)


if __name__ == "__main__":
    main()