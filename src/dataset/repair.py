import datetime

from .store import Run, read_jsonl, write_jsonl


def marked_requests(samples: list[dict]) -> list[str]:
    marked = sorted({s["request_id"] for s in samples if s.get("regenerate")})
    if not marked and not any("regenerate" in s for s in samples):
        raise ValueError("samples carry no regenerate flag; run `analyze` first")
    return marked


def repair(
    run: Run,
    kind: str,
    *,
    request_ids=None,
    sample_ids=None,
    drop_plan=False,
) -> dict:
    samples = read_jsonl(run.samples(kind))
    if not samples:
        raise ValueError(f"no samples for {kind}")

    targets = set(request_ids or [])
    if sample_ids:
        targets |= {s["request_id"] for s in samples if s["id"] in set(sample_ids)}
    if not request_ids and not sample_ids:
        targets = set(marked_requests(samples))

    if not targets:
        return {"removed": 0, "kept": len(samples), "removed_ids": [], "backup": None}

    keep = [s for s in samples if s["request_id"] not in targets]
    removed_ids = [s["id"] for s in samples if s["request_id"] in targets]

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = run.samples(kind).with_name(f"samples.{stamp}.bak.jsonl")
    backup.write_bytes(run.samples(kind).read_bytes())
    write_jsonl(run.samples(kind), keep)

    plan_dropped = 0
    if drop_plan:
        plan_rows = read_jsonl(run.plan)
        keep_plan = [
            r for r in plan_rows if r["kind"] != kind or r["request_id"] not in targets
        ]
        plan_dropped = len(plan_rows) - len(keep_plan)
        run.plan.with_name(f"plan.{stamp}.bak.jsonl").write_bytes(run.plan.read_bytes())
        write_jsonl(run.plan, keep_plan)

    return {
        "removed": len(removed_ids),
        "kept": len(keep),
        "removed_requests": sorted(targets),
        "removed_ids": removed_ids,
        "backup": backup,
        "plan_rows_dropped": plan_dropped,
    }
