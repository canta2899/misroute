from . import checks, config, llm, prompts
from .providers import Spec
from .store import Run, read_jsonl, write_json, write_jsonl

# the sample fields the oracle owns
VERDICT_FIELDS = (
    "oracle_predicted",
    "oracle_correct",
    "oracle_decidable",
    "oracle_quote",
    "oracle_constraints",
    "oracle_model",
)


def audit(
    run: Run,
    kind: str,
    *,
    mode: str = "batch",
    limit=None,
    ids=None,
    redo=False,
) -> dict:
    model = config.ORACLE_MODEL

    rows = read_jsonl(run.samples(kind))
    samples = {s["id"]: s for s in rows}
    if not samples:
        raise ValueError(f"no samples for {kind}; generate first")

    key = VERDICT_FIELDS[0]
    if redo:
        # A hand-edited sample keeps its old verdict, which now describes text
        # that no longer exists. Clearing it makes the sample unjudged again.
        stale = set(ids) if ids else set(samples)
        for s in rows:
            if s["id"] in stale:
                for field in VERDICT_FIELDS:
                    s.pop(field, None)
        write_jsonl(run.samples(kind), rows)

    done = {sid for sid, s in samples.items() if s.get(key) is not None}

    todo = [
        s for sid, s in samples.items() if sid not in done and (not ids or sid in ids)
    ]
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"requested": 0, "written": 0, "failures": [], "already": len(done)}

    specs = {s["id"]: _spec(s) for s in todo}
    results = llm.run(
        model, specs, mode=mode, log_dir=run.llm_dir(kind), tag=f"{kind}_oracle"
    )
    outputs, failures = llm.split_results(results)

    written = 0
    for sample_id, output in outputs.items():
        predicted = output.get("expert")
        if predicted not in config.EXPERT_IDS:
            failures.append({"id": sample_id, "error": f"bad expert: {predicted!r}"})
            continue
        samples[sample_id].update(
            oracle_predicted=predicted,
            oracle_correct=predicted == samples[sample_id]["gold"],
            oracle_decidable=output.get("decidable", True),
            oracle_quote=output.get("quote"),
            oracle_constraints=output.get("constraints_present"),
            oracle_model=model,
        )
        written += 1
    write_jsonl(run.samples(kind), rows)
    return {
        "requested": len(todo),
        "written": written,
        "failures": failures,
        "already": len(done),
    }


def _spec(sample: dict) -> Spec:
    text = (
        checks.transcript(sample)
        if sample["kind"] == "multi_turn"
        else checks.user_text(sample)
    )
    return Spec(
        system=prompts.ORACLE_SYSTEM,
        user=prompts.audit_user(text),
        tool=prompts.ORACLE_TOOL,
        max_tokens=config.MAX_TOKENS["audit"],
    )


def summarize(run: Run, kind: str) -> dict:
    samples = read_jsonl(run.samples(kind))
    report = {}
    judged = [s for s in samples if s.get("oracle_predicted")]
    if judged:
        report["oracle"] = _accuracy_block(judged)

    report["disputed"] = sorted(
        s["id"] for s in samples if s.get("review_decision") == "valid"
    )
    report["unjudged"] = sorted(
        s["id"] for s in samples if not s.get("oracle_predicted")
    )
    write_json(run.reports(kind) / "audit.json", report)
    return report


def _accuracy_block(judged: list[dict]) -> dict:
    correct, predicted = "oracle_correct", "oracle_predicted"

    def acc(subset):
        return (
            round(sum(s[correct] for s in subset) / len(subset), 4) if subset else None
        )

    by_construction = {}
    for construction in config.CONSTRUCTION_IDS:
        subset = [s for s in judged if s["construction"] == construction]
        if subset:
            by_construction[construction] = {"n": len(subset), "accuracy": acc(subset)}

    confusion: dict[str, dict[str, int]] = {}
    for s in judged:
        confusion.setdefault(s["gold"], {}).setdefault(s[predicted], 0)
        confusion[s["gold"]][s[predicted]] += 1

    return {
        "n": len(judged),
        "model": judged[0]["oracle_model"],
        "accuracy": acc(judged),
        "by_construction": by_construction,
        "n_undecidable": sum(1 for s in judged if not s.get("oracle_decidable", True)),
        "confusion": confusion,
    }
