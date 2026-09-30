"""Fine-tune ModernBERT-large as a 5-way sub-agent classifier, out of fold.

    uv run modal run -m simulate.modernbert_app
    uv run modal run -m simulate.modernbert_app --kinds single_turn --folds 0 --epochs 1   # smoke test
    uv run modal run -m simulate.modernbert_app --seeds 0,1,2

The strongest text-only baseline: the encoder and a linear head on 5 classes train
together on our labels (cross-entropy), so the text alone gets every chance the probe
gets and more (the probe's LLM never trains on our labels).

Input: the `turns` column rendered as `role: content` blocks, the conversation the
coordinator sees without its system prompt. MAX_LEN covers every sample; anything
longer is cut from the left, so the last turn survives (`truncated` in the csv).

Folds: the scenario-grouped `GroupKFold(5)` over dataset/<kind>.csv, the same call as
`analysis.probe.folds_for`, so the same folds as the probes and the baselines. One
container per (kind, fold, seed): train on the other 4 folds, predict fold k.

Recipe: fixed, never tuned on a test fold. AdamW, LR 3e-5 (inside the ModernBERT
paper's GLUE grid), weight decay 0.01, 10% linear warmup then linear decay, EPOCHS
epochs, effective batch 16, bf16 autocast, grad clip 1.0.

Volume `modernbert-finetune` at /cache:
    /cache/hf                                 HF_HOME: the base checkpoint
    /cache/ft/<kind>/fold<k>/seed<s>/         the fine-tuned model + tokenizer (bf16)

Output (local): <out-dir>/<kind>.csv, one row per (seed, sample) with the fold,
the prediction and p_<expert> for every expert.
"""

import csv
import json
import os
import random
from pathlib import Path

import modal

BASE = "answerdotai/ModernBERT-large"
MODEL = "modernbert-large-ft"
EXPERTS = (  # label index order, as analysis.probe.EXPERTS
    "time_to_market",
    "scale_arch",
    "resource_opt",
    "legacy_compat",
    "strict_compliance",
)
KINDS = ("single_turn", "multi_turn")
N_FOLDS = 5

MAX_LEN = 1024  # longest sample: 257 tokens single_turn, 797 multi_turn
EPOCHS = 5
MICRO_BATCH = 8
GRAD_ACCUM = 2  # effective batch 16
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP = 0.1
EVAL_BATCH = 32

GPU = os.environ.get("GPU_TYPE", "L4")
DATASET_DIR = "/root/dataset"
CACHE = "/cache"
ENV = {"HF_HOME": f"{CACHE}/hf"}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.12.0", "transformers>=5.14.1", "scikit-learn", "joblib"
    )  # sklearn, joblib: modernbert_frozen_app
    .add_local_dir("dataset", DATASET_DIR, ignore=["merged"])
)
volume = modal.Volume.from_name("modernbert-finetune", create_if_missing=True)

app = modal.App("modernbert-finetune")


def render(turns: str) -> str:
    return "\n\n".join(f"{t['role']}: {t['content']}" for t in json.loads(turns))


@app.function(image=image, env=ENV, volumes={CACHE: volume}, timeout=30 * 60)
def fetch_base():
    """Download once, before the folds start, so no two containers write the cache."""
    from huggingface_hub import snapshot_download

    snapshot_download(BASE)
    volume.commit()


@app.function(
    image=image, gpu=GPU, env=ENV, volumes={CACHE: volume}, timeout=2 * 60 * 60
)
def finetune(
    kind: str, fold: int, seed: int, train_ids: list, test_ids: list, epochs: int
) -> list[dict]:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")

    with open(f"{DATASET_DIR}/{kind}.csv", newline="") as f:
        by_id = {s["id"]: s for s in csv.DictReader(f)}
    train = [by_id[i] for i in train_ids]
    test = [by_id[i] for i in test_ids]

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.truncation_side = "left"  # keep the last turn
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE,
        num_labels=len(EXPERTS),
        id2label=dict(enumerate(EXPERTS)),
        label2id={e: i for i, e in enumerate(EXPERTS)},
    ).to(device)

    def encode(samples):
        texts = [render(s["turns"]) for s in samples]
        full = tok(texts, add_special_tokens=True)["input_ids"]
        batch = tok(
            texts,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        return batch, [len(ids) for ids in full]

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_micro = -(-len(train) // MICRO_BATCH)
    total = -(-n_micro // GRAD_ACCUM) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * total), total)
    print(f"[ft] {kind} fold {fold} seed {seed}: {len(train)} train, {len(test)} test")

    model.train()
    for epoch in range(epochs):
        random.shuffle(train)
        epoch_loss, hits = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, start in enumerate(range(0, len(train), MICRO_BATCH)):
            chunk = train[start : start + MICRO_BATCH]
            batch, _ = encode(chunk)
            labels = torch.tensor([EXPERTS.index(s["gold"]) for s in chunk])
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(**batch.to(device)).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels.to(device))
            (loss / GRAD_ACCUM).backward()
            if (step + 1) % GRAD_ACCUM == 0 or step + 1 == n_micro:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * len(chunk)
            hits += (logits.argmax(-1).cpu() == labels).sum().item()
        print(
            f"[ft] {kind} fold {fold} seed {seed} epoch {epoch + 1}/{epochs}: "
            f"loss={epoch_loss / len(train):.4f} train_acc={hits / len(train):.3f}"
        )

    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(test), EVAL_BATCH):
            chunk = test[start : start + EVAL_BATCH]
            batch, n_tokens = encode(chunk)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(**batch.to(device)).logits
            probs = torch.softmax(logits.float(), -1).cpu().tolist()
            for s, p, n in zip(chunk, probs, n_tokens):
                pred = EXPERTS[max(range(len(p)), key=p.__getitem__)]
                rows.append(
                    {
                        "id": s["id"],
                        "kind": kind,
                        "model": MODEL,
                        "fold": fold,
                        "seed": seed,
                        "construction": s["construction"],
                        "gold": s["gold"],
                        "predicted": pred,
                        "correct": pred == s["gold"],
                        **{f"p_{e}": v for e, v in zip(EXPERTS, p)},
                        "n_tokens": n,
                        "truncated": n > MAX_LEN,
                    }
                )
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"[ft] {kind} fold {fold} seed {seed}: test acc={acc:.3f}")

    out = f"{CACHE}/ft/{kind}/fold{fold}/seed{seed}"
    model.to(torch.bfloat16).save_pretrained(out)
    tok.save_pretrained(out)
    volume.commit()
    return rows


@app.local_entrypoint()
def main(
    kinds: str = ",".join(KINDS),
    folds: str = ",".join(map(str, range(N_FOLDS))),
    seeds: str = "0",
    epochs: int = EPOCHS,
    out_dir: str = "data/sim/modernbert",
):
    import numpy as np
    from sklearn.model_selection import GroupKFold

    wanted = {int(k) for k in folds.split(",")}
    jobs = []
    for kind in kinds.split(","):
        with open(f"dataset/{kind}.csv", newline="") as f:
            samples = list(csv.DictReader(f))
        ids = np.array([s["id"] for s in samples])
        groups = np.array([s["scenario_id"] for s in samples])
        y = np.array([s["gold"] for s in samples])
        # analysis.probe.folds_for: the same call, so the same folds
        for k, (train_idx, test_idx) in enumerate(
            GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, groups)
        ):
            if k in wanted:
                for seed in map(int, seeds.split(",")):
                    jobs.append(
                        (kind, k, seed, ids[train_idx].tolist(), ids[test_idx].tolist(), epochs)
                    )  # fmt: skip

    fetch_base.remote()
    results = list(finetune.starmap(jobs, return_exceptions=True))

    rows = []
    for job, res in zip(jobs, results):
        if isinstance(res, BaseException):
            print(f"[ft] FAILED {job[0]} fold {job[1]} seed {job[2]}: {res!r}")
        else:
            rows += res

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for kind in sorted({r["kind"] for r in rows}):
        part = [r for r in rows if r["kind"] == kind]
        path = out / f"{kind}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(part[0]))
            writer.writeheader()
            writer.writerows(part)
        acc = sum(r["correct"] for r in part) / len(part)
        print(f"[ft] wrote {len(part)} rows -> {path} (pooled acc={acc:.3f})")
