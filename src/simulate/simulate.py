import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from simulate.models.base import BaseModel

HIDDEN_DTYPE = np.float16  # hidden states are stored as fp16


def run_simulation(
    model: BaseModel,
    input_path: str | Path,
    out_dir: str | Path,
    *,
    batch_size: int,
    shard_size: int,
    limit: int | None = None,
    on_shard: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # may clash with existing shards, if you re-run a simulation
    # make sure to remove stale shards from the volume
    stale = sorted(out_dir.glob("shard_*.parquet"))
    for path in stale:
        path.unlink()

    samples = load_samples(input_path)
    if limit:
        samples = samples[:limit]

    pending = []  # rows not yet flushed to a shard
    shard_idx = 0
    n_seen = n_correct = n_unrouted = 0

    if stale:
        print(f"[sim] cleared {len(stale)} shard(s) left by an earlier run")
    print(
        f"[sim] model={model.model_name} | {len(samples)} samples | "
        f"batch={batch_size} shard={shard_size} "
        f"max_new_tokens={model.max_new_tokens} tkw={model.template_kwargs}\n"
        f"[sim] gen={model.gen_kwargs}\n"
        f"[sim] writing shards -> {out_dir}"
    )

    bar = tqdm(total=len(samples), unit="sample", dynamic_ncols=True)
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        rows = run_batch(model, batch)

        for r in rows:
            n_seen += 1
            if r["predicted_expert"] is None:
                n_unrouted += 1
            elif r["correct"]:
                n_correct += 1
        pending.extend(rows)

        bar.update(len(batch))
        bar.set_postfix(acc=f"{n_correct / max(n_seen, 1):.3f}", unrouted=n_unrouted)

        while len(pending) >= shard_size:
            path = write_shard(pending[:shard_size], out_dir, shard_idx)
            pending = pending[shard_size:]
            shard_idx += 1
            if on_shard is not None:
                on_shard(path)

    if pending:
        path = write_shard(pending, out_dir, shard_idx)
        if on_shard is not None:
            on_shard(path)
    bar.close()

    acc = n_correct / max(n_seen, 1)
    print(
        f"[sim] done: {n_seen} samples | routing acc={acc:.4f} | "
        f"unrouted={n_unrouted} | {shard_idx + (1 if pending else 0)} shards in {out_dir}"
    )
    return {
        "n": n_seen,
        "accuracy": acc,
        "unrouted": n_unrouted,
        "out_dir": str(out_dir),
    }


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path).to_dict("records")  # type: ignore
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_shard(
    rows: list[dict[str, Any]], out_dir: str | Path, shard_idx: int
) -> Path:
    path = Path(out_dir) / f"shard_{shard_idx:05d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    tqdm.write(f"[sim] wrote {path.name} ({len(rows)} samples)")
    return path


def sample_messages(sample: dict[str, Any]) -> list[dict]:
    turns = sample.get("turns")
    if isinstance(turns, str):
        turns = json.loads(turns)
    if turns:
        return turns
    conversation = sample.get("conversation")
    if conversation:
        return conversation
    return [{"role": "user", "content": sample["query"]}]


def sample_query_text(sample: dict[str, Any]) -> str | None:
    """The text being routed, for the stored row: the last user turn."""
    user_turns = [t for t in sample_messages(sample) if t.get("role") == "user"]
    return user_turns[-1]["content"] if user_turns else None


def run_batch(model: BaseModel, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run one batch of samples, returning a list of result-row dicts."""
    routed = model.generate([sample_messages(s) for s in batch])
    responses, hidden = routed.responses, routed.hidden
    n_layers, _, hidden_dim = hidden.shape

    rows = []
    for i, sample in enumerate(batch):
        predicted = model.parse(responses[i])
        gold = sample.get("gold") or sample.get("gold_expert")
        vec = hidden[:, i, :].astype(HIDDEN_DTYPE)  # [n_layers, hidden_dim]
        rows.append(
            {
                "id": sample["id"],
                "gold_expert": gold,
                "predicted_expert": predicted,
                "correct": predicted == gold,
                "query": sample_query_text(sample),
                "raw_response": responses[i],
                "hidden_states": vec.tobytes(),
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
                "hidden_dtype": np.dtype(HIDDEN_DTYPE).name,
            }
        )
    return rows
