import json, os, itertools, threading, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import make_pair_instances
from .manifest import write_dataset, DATA_ROOT

RAW = DATA_ROOT / "raw"

# The model emits typographic unicode (non-breaking hyphens, en/em dashes, narrow
# spaces) in the SENTENCE while writing plain ASCII in the entity list so
# sentence.find(entity) fails and the sample is rejected. Normalise both sides
# identically before matching. Markdown bold is stripped for the same reason.
_UNICODE_FIXES = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u200b": "",
}


def _normalise(s):
    for a, b in _UNICODE_FIXES.items():
        s = s.replace(a, b)
    s = s.replace("**", "").replace("__", "")        # markdown emphasis
    s = unicodedata.normalize("NFKC", s)             # also folds CH₃ -> CH3
    return " ".join(s.split())                       # collapse whitespace


def _is_degenerate(text, max_run=6, min_unique=0.3):
    """Catch the 'mandatory mandatory mandatory...' decoding loops."""
    words = text.split()
    if len(words) < 15:
        return False
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        if run > max_run:
            return True
    return len(set(w.lower() for w in words)) / len(words) < min_unique


# shims that quack like bioc's BratDocument

class _Loc:
    def __init__(self, b, e): self._b, self._e = b, e
    def begin(self): return self._b
    def end(self): return self._e


class SynthEntity:
    def __init__(self, id, begin, end, type="drug"):
        self.id, self.locations, self.type = id, _Loc(begin, end), type


class SynthRelation:
    def __init__(self, arg1, arg2, type):
        self.arguments, self.type = {"Arg1": arg1, "Arg2": arg2}, type


class SynthDoc:
    def __init__(self, text, entities, relations, register, sent_id):
        self.text, self.entities, self.relations = text, entities, relations
        self.register, self.sent_id = register, sent_id


class Rejected(Exception):
    """Raised when a generated sample can't be turned into valid instances."""


def _find_nth(sentence, surface, n):
    """Index of the (n+1)-th occurrence of `surface`, or -1.

    Exact match first. Falls back to case-insensitive, because the model
    routinely capitalises a name at the start of a sentence ("antiplatelet
    agents" -> "Antiplatelet agents") or lower-cases one mid-sentence
    ("Quinacrine diHCl" -> "quinacrine diHCl"). The span length is unchanged
    either way, so the offsets stay valid.
    """
    for hay, needle in ((sentence, surface), (sentence.lower(), surface.lower())):
        start = -1
        for _ in range(n + 1):
            start = hay.find(needle, start + 1)
            if start == -1:
                break
        if start != -1:
            return start
    return -1


def resolve_spans(sentence, entities):
    """Locate each declared entity mention in the sentence by counting occurrences.

    entities: list of {"id": "T1", "text": "ORENCIA", "type": "brand"} in order of
    appearance. The model says WHICH mentions exist and in what order; we compute
    WHERE (models cannot count characters reliably).

    Mirrors the .ann convention: a surface form appearing twice is two entities
    with distinct ids and distinct offsets.
    """
    seen = {}          # surface text (lowercased) -> occurrences already consumed
    resolved = []
    for ent in entities:
        surface = ent["text"]
        if not surface:
            raise Rejected("empty entity text")
        key = surface.lower()
        n = seen.get(key, 0)
        start = _find_nth(sentence, surface, n)
        if start == -1:
            raise Rejected(
                f"entity {ent['id']!r} ({surface!r}) occurrence #{n + 1} not found in sentence"
            )
        seen[key] = n + 1
        resolved.append(SynthEntity(ent["id"], start, start + len(surface),
                                    ent.get("type", "drug")))

    # overlapping spans mean the marker insertion would produce garbage
    ordered = sorted(resolved, key=lambda e: e.locations.begin())
    for a, b in zip(ordered, ordered[1:]):
        if b.locations.begin() < a.locations.end():
            raise Rejected(f"overlapping entity spans: {a.id} and {b.id}")
    return resolved


def sample_to_instances(sample, sent_id, register="synthetic", max_words=120):
    """One raw model sample -> list of {text,label,source,sent_id} instances.

    Every entity pair is enumerated; pairs absent from `relations` fall through to
    NONE. That is where in-distribution negatives come from -- for free.
    """
    if not isinstance(sample, dict):
        raise Rejected("no sample returned")

    sentence = _normalise(sample.get("sentence") or "")
    if not sentence:
        raise Rejected("empty sentence")
    if len(sentence.split()) > max_words:
        raise Rejected(f"sentence too long ({len(sentence.split())} words)")
    if _is_degenerate(sentence):
        raise Rejected("degenerate repetition in sentence")

    entities = sample.get("entities") or []
    if len(entities) < 2:
        raise Rejected(f"need >=2 entities to form a pair, got {len(entities)}")

    # normalise entity surface forms the same way, and reject the failure mode
    # where the model puts the whole sentence in the entity text field
    norm_entities = []
    for e in entities:
        text = _normalise(e.get("text") or "")
        if len(text.split()) > 12 or text == sentence:
            raise Rejected("entity text looks like a sentence, not a name")
        norm_entities.append({**e, "text": text})

    resolved = resolve_spans(sentence, norm_entities)

    # Relation args are referenced inconsistently: sometimes by the declared id,
    # sometimes by the entity text, and sometimes the model fills every id with a
    # literal like "text". Build an unambiguous lookup; deliberately NO positional
    # fallback, because 0-based vs 1-based cannot be told apart and guessing wrong
    # silently mislabels the pair.
    canon = [f"T{i}" for i in range(len(resolved))]
    for e, c in zip(resolved, canon):
        e.id = c

    lookup = {}
    declared = [str(e.get("id", "")) for e in norm_entities]
    if len(set(declared)) == len(declared):          # ids are usable only if unique
        lookup.update(dict(zip(declared, canon)))
    for e, c in zip(norm_entities, canon):           # ...else fall back to the text
        lookup.setdefault(e["text"].lower(), c)

    valid_ids = set(canon)
    relations = []
    for rel in sample.get("relations") or []:
        a1 = lookup.get(str(rel.get("arg1_id")), lookup.get(str(rel.get("arg1_id", "")).lower()))
        a2 = lookup.get(str(rel.get("arg2_id")), lookup.get(str(rel.get("arg2_id", "")).lower()))
        if a1 not in valid_ids or a2 not in valid_ids:
            raise Rejected(f"relation references unresolvable entity: "
                           f"{rel.get('arg1_id')!r}, {rel.get('arg2_id')!r}")
        if a1 == a2:
            raise Rejected("self-relation")
        relations.append(SynthRelation(a1, a2, rel["label"]))

    doc = SynthDoc(sentence, resolved, relations, register, sent_id)
    return make_pair_instances(doc)


def generate_raw(specs, sample_fn, gen_id, max_workers=64, resume=True):
    """Call sample_fn(spec) concurrently; append each result to raw/<gen_id>.jsonl.

    Appends as results arrive, so a dead pod costs only in-flight requests.
    resume=True skips specs already present in the raw file.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{gen_id}.jsonl"

    done = set()
    if resume and path.exists():
        for line in path.read_text().splitlines():
            if line:
                rec = json.loads(line)
                # only successes count as done -- errored specs must be retried,
                # otherwise transient API failures silently shrink the dataset
                if not rec.get("error"):
                    done.add(rec["spec_index"])
        print(f"resuming {gen_id}: {len(done)} already succeeded")

    todo = [(i, s) for i, s in enumerate(specs) if i not in done]
    if not todo:
        print("nothing to do")
        return path

    lock = threading.Lock()
    n_ok = n_err = 0

    def _work(item):
        i, spec = item
        try:
            return i, spec, sample_fn(spec), None
        except Exception as e:                      # network/API/parse failure
            return i, spec, None, f"{type(e).__name__}: {e}"

    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(todo), desc=f"gen {gen_id}", unit="req", smoothing=0.1)
    except ImportError:
        bar = None

    with open(path, "a") as fp, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, item) for item in todo]
        for fut in as_completed(futures):
            i, spec, sample, err = fut.result()
            with lock:
                fp.write(json.dumps({"spec_index": i, "spec": spec,
                                     "sample": sample, "error": err}) + "\n")
                fp.flush()                          # survive an abrupt pod death
                if err: n_err += 1
                else:   n_ok += 1
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(ok=n_ok, err=n_err, refresh=False)
                elif (n_ok + n_err) % 200 == 0:
                    print(f"  {n_ok + n_err}/{len(todo)}  (errors: {n_err})")
    if bar is not None:
        bar.close()

    print(f"stage 1 done: {n_ok} ok, {n_err} api/parse errors -> {path}")
    return path


def build_dataset_from_raw(gen_id, generator, vocab_source=None,
                           negative_strategy=None, seed=None, notes=""):
    """Deterministic: raw model output -> instances -> manifested dataset.

    Free to re-run whenever the resolver or pair logic changes. Returns
    (dataset_id, stats) where stats records exactly what was thrown away and why.
    """
    path = RAW / f"{gen_id}.jsonl"
    instances, rejects = [], []
    n_api_err = 0

    for line in path.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("error"):
            n_api_err += 1
            continue
        sent_id = f"synth:{gen_id}:{rec['spec_index']}"
        register = (rec.get("spec") or {}).get("register", "synthetic")
        try:
            instances.extend(sample_to_instances(rec["sample"], sent_id, register=register))
        except Rejected as e:
            rejects.append({"spec_index": rec["spec_index"], "reason": str(e),
                            "sample": rec["sample"]})

    n_samples = len(instances) and len({r["sent_id"] for r in instances})
    stats = {
        "gen_id": gen_id,
        "n_api_errors": n_api_err,
        "n_rejected": len(rejects),
        "n_sentences_used": n_samples,
        "n_instances": len(instances),
        "reject_reasons": _reason_counts(rejects),
    }

    # keep the rejects: they are the diagnostic for prompt iteration
    if rejects:
        (RAW / f"{gen_id}.rejects.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rejects) + "\n")

    if not instances:
        raise Rejected(f"no usable instances from {gen_id}: {stats}")

    dataset_id = write_dataset(
        instances, provenance="synthetic", generator={**(generator or {}), "gen_id": gen_id},
        vocab_source=vocab_source, negative_strategy=negative_strategy,
        seed=seed, notes=notes,
    )
    print(f"stage 2: {stats['n_rejected']} rejected, {stats['n_instances']} instances kept")
    return dataset_id, stats


def _reason_counts(rejects):
    out = {}
    for r in rejects:
        key = r["reason"].split(":")[0].split(" not found")[0][:60]
        out[key] = out.get(key, 0) + 1
    return out