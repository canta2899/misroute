import time

from tqdm import tqdm

from .providers import Result, Spec, get_provider
from .store import append_jsonl, write_json


def run(
    model: str,
    specs: dict[str, Spec],
    *,
    mode: str = "batch",
    log_dir=None,
    tag: str = "call",
    poll_seconds: int = 30,
) -> dict[str, Result]:
    if not specs:
        return {}
    provider = get_provider(model)
    if mode == "direct":
        results = _direct(provider, specs, tag)
    elif mode == "batch":
        results = _batch(provider, specs, tag, log_dir, poll_seconds)
    else:
        raise ValueError(f"mode must be 'batch' or 'direct', got {mode!r}")
    _log_raw(log_dir, tag, results)
    return results


def _direct(provider, specs, tag) -> dict[str, Result]:
    out = {}
    for cid, spec in tqdm(specs.items(), desc=f"{tag} (direct)"):
        try:
            out[cid] = provider.call(spec)
        except Exception as exc:  # keep going; the id stays a gap in the plan
            out[cid] = Result(raw={}, error=f"{type(exc).__name__}: {exc}")
    return out


def _batch(provider, specs, tag, log_dir, poll_seconds) -> dict[str, Result]:
    batch_id = provider.submit_batch(list(specs.items()))
    print(f"submitted batch {batch_id} ({len(specs)} requests)")
    if log_dir is not None:
        write_json(
            log_dir / "batches" / f"{batch_id}.json",
            {
                "batch_id": batch_id,
                "tag": tag,
                "model": provider.model,
                "custom_ids": list(specs),
            },
        )
    while True:
        status = provider.poll_batch(batch_id)
        if status.done:
            break
        print(f"[{batch_id}] {status.counts}", flush=True)
        time.sleep(poll_seconds)

    fetched = provider.fetch_batch(batch_id)
    return {
        cid: fetched.get(cid, Result(raw={}, error="no result returned"))
        for cid in specs
    }


def _log_raw(log_dir, tag, results: dict[str, Result]) -> None:
    if log_dir is None:
        return
    append_jsonl(
        log_dir / "raw" / f"{tag}.jsonl",
        [
            {"custom_id": cid, "error": r.error, "response": r.raw}
            for cid, r in results.items()
        ],
    )


def split_results(results: dict[str, Result]) -> tuple[dict, list]:
    ok, failures = {}, []
    for cid, result in results.items():
        if result.error or result.output is None:
            failures.append({"id": cid, "error": result.error or "empty output"})
        else:
            ok[cid] = result.output
    return ok, failures


__all__ = ["run", "split_results", "Spec", "Result"]
