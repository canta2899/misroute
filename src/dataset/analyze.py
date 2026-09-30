from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from . import checks, config
from .store import Run, read_jsonl, write_json, write_jsonl

_embedder = None

_KEEP = ("valid", "fixed", "accepted")


def embed(texts: list[str]) -> np.ndarray:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        # a handful of multi-turn transcripts exceed the 512-token window; the
        # deciding turn is the last one, so drop from the front, not the back
        _embedder.tokenizer.truncation_side = "left"
    return np.asarray(
        _embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=len(texts) > 256
        )
    )


def _logreg():
    return LogisticRegression(max_iter=2000, random_state=config.SEED)


def _accuracy(labels, pred, groups, want=None):
    idx = [i for i in range(len(labels)) if want is None or groups[i] == want]
    if not idx:
        return None
    return round(
        float(accuracy_score([labels[i] for i in idx], [pred[i] for i in idx])), 4
    )


def _cv_predict(estimator, features, labels, groups):
    n_splits = min(5, len(set(groups)), min(Counter(labels).values()))
    if n_splits < 2:
        return None
    return cross_val_predict(
        estimator, features, labels, cv=GroupKFold(n_splits=n_splits), groups=groups
    )


def baselines(samples: list[dict], text_fn) -> dict:
    texts = [text_fn(s) for s in samples]
    labels = [s["gold"] for s in samples]
    groups = [s["scenario_id"] for s in samples]
    constructions = [s["construction"] for s in samples]

    models = {
        "lexical_bow": (
            make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=2), _logreg()),
            texts,
        ),
        "lexical_tfidf": (
            make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), _logreg()),
            texts,
        ),
        "embedding": (_logreg(), None),
    }

    report = {
        "n": len(samples),
        "chance": round(config.CHANCE, 4),
        "n_by_construction": dict(Counter(constructions)),
    }
    for name, (estimator, features) in models.items():
        if features is None:
            features = embed(texts)
        pred = _cv_predict(estimator, features, labels, groups)
        if pred is None:
            report[name] = None
            continue
        block = {"overall": _accuracy(labels, pred, constructions)}
        for c in config.CONSTRUCTION_IDS:
            block[c] = _accuracy(labels, pred, constructions, c)
        report[name] = block
    return report


def _duplicate_map(samples: list[dict]) -> tuple[dict, dict]:
    by_request: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_request[s["request_id"]].append(s)

    strata: dict[tuple, list[str]] = defaultdict(list)
    for request_id, members in by_request.items():
        head = members[0]
        strata[(head["gold"], head["construction"])].append(request_id)

    dup_of, dup_sim = {}, {}
    for _, request_ids in sorted(strata.items()):
        request_ids.sort()
        if len(request_ids) < 2:
            continue
        vectors = embed([checks.user_text(by_request[r][0]) for r in request_ids])
        kept: list[int] = []
        for i in range(len(request_ids)):
            if not kept:
                kept.append(i)
                continue
            sims = vectors[kept] @ vectors[i]
            best = int(np.argmax(sims))
            if float(sims[best]) > config.DEDUP_COSINE_THRESHOLD:
                twin = by_request[request_ids[kept[best]]][0]["id"]
                for member in by_request[request_ids[i]]:
                    dup_of[member["id"]] = twin
                    dup_sim[member["id"]] = round(float(sims[best]), 4)
            else:
                kept.append(i)
    return dup_of, dup_sim


def analyze(run: Run, kind: str) -> dict:
    samples = read_jsonl(run.samples(kind))
    if not samples:
        raise ValueError(f"no samples for {kind}; generate first")
    reports_dir = run.reports(kind)

    annotate(samples)
    write_jsonl(run.samples(kind), samples)

    valid = [s for s in samples if s["structural_ok"]]
    views = {"user_turns": checks.user_text}
    if kind == "multi_turn":
        views["full_transcript"] = checks.transcript
    r_baselines = {name: baselines(valid, fn) for name, fn in views.items()}

    _, r_pairs = checks.pair_overlaps(valid)
    r_dup = _duplicate_report(samples)
    r_struct = _structural_report(samples)

    for name, report in (
        ("baselines", r_baselines),
        ("pairs", r_pairs),
        ("duplicates", r_dup),
        ("structure", r_struct),
    ):
        write_json(reports_dir / f"{name}.json", report)

    return {
        "n_samples": len(samples),
        "n_valid": len(valid),
        "baselines": r_baselines,
        "pairs": r_pairs,
        "reasons": _reason_counts(samples),
        "n_regenerate": sum(s["regenerate"] for s in samples),
        "reports_dir": reports_dir,
    }


def annotate(samples: list[dict]) -> None:
    problems_by_id = {s["id"]: checks.problems(s) for s in samples}
    dup_of, dup_sim = _duplicate_map(samples)
    overlaps = _pair_overlaps_by_id(samples)
    identical = _identical_pair_ids(samples)

    for s in samples:
        errors = problems_by_id[s["id"]]
        s["structural_ok"] = not errors
        s["structural_errors"] = errors
        s["duplicate_of"] = dup_of.get(s["id"])
        s["duplicate_similarity"] = dup_sim.get(s["id"])
        s["pair_overlap"] = overlaps.get(s["request_id"])
        s["pair_identical"] = s["request_id"] in identical or None
        s.setdefault("review_decision", None)
        s.setdefault("review_note", None)

    flagged = {s["id"]: bool(_reasons(s)) for s in samples}
    pairs: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        if s["construction"] == "adversarial":
            pairs[s["request_id"]].append(s)
    for members in pairs.values():
        # a member the human cleared never drags its twin into regeneration
        raised = [
            m
            for m in members
            if flagged[m["id"]] and m.get("review_decision") not in _KEEP
        ]
        if len(members) == 2 and raised:
            for m in members:
                flagged[m["id"]] = True

    for s in samples:
        decision = s.get("review_decision")
        if decision in _KEEP:
            s["regenerate"] = 0
        elif decision == "regenerate":
            s["regenerate"] = 1
        else:
            s["regenerate"] = int(flagged[s["id"]])


def _reasons(s: dict) -> list[str]:
    """Why this sample would be regenerated. Empty means it is fine."""
    out = []
    if not s["structural_ok"]:
        out.append("invalid")
    if s.get("duplicate_of"):
        out.append("duplicate")
    if s.get("pair_identical"):
        out.append("identical_pair")
    elif s.get("pair_overlap") is not None and s[
        "pair_overlap"
    ] < checks.min_pair_overlap(s["kind"]):
        out.append("weak_pair")
    if s.get("oracle_predicted"):
        if not s.get("oracle_decidable", True):
            out.append("undecidable")
        elif not s["oracle_correct"]:
            out.append("oracle_disagreement")
    else:
        out.append("unjudged")
    return out


def _reason_counts(samples: list[dict]) -> dict:
    counts = Counter(
        reason for s in samples if s["regenerate"] for reason in _reasons(s)
    )
    return {
        k: counts.get(k, 0)
        for k in (
            "invalid",
            "duplicate",
            "weak_pair",
            "identical_pair",
            "undecidable",
            "oracle_disagreement",
            "unjudged",
        )
    }


def _identical_pair_ids(samples: list[dict]) -> set:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        if s["construction"] == "adversarial":
            pairs[s["request_id"]].append(s)
    return {
        rid
        for rid, ms in pairs.items()
        if len(ms) == 2
        and checks.contrast_text(ms[0]).strip() == checks.contrast_text(ms[1]).strip()
    }


def _pair_overlaps_by_id(samples: list[dict]) -> dict:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        if s["construction"] == "adversarial":
            pairs[s["request_id"]].append(s)
    out = {}
    for request_id, members in pairs.items():
        out[request_id] = (
            round(
                checks.overlap(
                    checks.contrast_text(members[0]), checks.contrast_text(members[1])
                ),
                4,
            )
            if len(members) == 2
            else 0.0
        )
    return out


def _structural_report(samples: list[dict]) -> dict:
    failed = [s for s in samples if not s["structural_ok"]]
    return {
        "n": len(samples),
        "n_ok": len(samples) - len(failed),
        "n_failed": len(failed),
        "errors": dict(Counter(e for s in failed for e in s["structural_errors"])),
    }


def _duplicate_report(samples: list[dict]) -> dict:
    dups = [s for s in samples if s.get("duplicate_of")]
    return {
        "threshold": config.DEDUP_COSINE_THRESHOLD,
        "n_duplicate_samples": len(dups),
        "per_stratum": dict(Counter(f"{s['gold']}/{s['construction']}" for s in dups)),
    }
