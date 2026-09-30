"""Fine-tune laya `multilingual` on single_turn with RLCD, out of fold.

    uv run modal run -m simulate.laya_finetune_app
    uv run modal run -m simulate.laya_finetune_app --folds 0 --epochs 1   # smoke test

The training loop is `train_ddp.py` from laya's notebook
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) on one GPU: RLCD
(Reinforcement Learning for Calibrated Decisions, a policy gradient whose reward is
the proper scoring rule `laya.common.proper_reward`) plus soft cross-entropy, then one
temperature fitted on a held-out calibration slice. Changes to that script:

    - one GPU, no DDP; GRAD_ACCUM doubles, so the effective batch stays 64
    - bf16 autocast (L4 / A10G) in place of fp16 with a grad scaler (T4)
    - the option order is shuffled per sample and epoch, so position cannot carry
      the label (the zero-shot checkpoints send most samples to the 2nd option)
    - the calibration slice is whole scenarios, like every other split here

Folds: the scenario-grouped `GroupKFold(5)` over dataset/single_turn.csv, the same
folds as the probes and the baselines. Run k trains on the other 4 folds minus a
calibration slice (CALIB_SHARE of their scenarios), fits T on the slice and predicts
fold k, so every sample is predicted once by a model that never saw its scenario.
The local entrypoint computes the splits; one container per fold.

The prompt is the best zero-shot variant, `handles|task|turns`, built by
`laya_app.build_variants`; budgets and the csv row are `laya_app`'s too.

Volume `laya-finetune` at /cache:
    /cache/hf                         HF_HOME: the base checkpoint
    /cache/ft/single_turn/fold<k>/    the fine-tuned checkpoint, T in rl_agent_config.json

Output (local): <out-dir>/single_turn.csv in the zero-shot layout (model
`laya-multilingual-ft`) plus `temperature`, and <out-dir>/splits.json.
"""

import csv
import json
import os
import random
from pathlib import Path

import modal

from simulate.laya_app import (
    BATCH_SIZE,
    CHECKPOINTS,
    DATASET_DIR,
    GPU,
    REPO,
    budgets,
    build_variants,
    result_row,
)
from simulate.laya_app import image as laya_image

CHECKPOINT = "multilingual"
KIND = "single_turn"
VARIANT = ("handles", "task", "turns")
MODEL = f"laya-{CHECKPOINT}-ft"
N_FOLDS = 5
CALIB_SHARE = 0.1  # of the training scenarios, held out for the temperature
SEED = 0

# train_ddp.py's recipe
EPOCHS = 4
MICRO_BATCH = 8
GRAD_ACCUM = 8  # 4 on 2 GPUs there
GROUP_SIZE = 4  # noisy samples per item, their mean reward is the baseline
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
SIGMA_START, SIGMA_END = 0.4, 0.1  # exploration noise on the logits

CACHE = "/cache"
volume = modal.Volume.from_name("laya-finetune", create_if_missing=True)
image = laya_image.add_local_python_source("simulate")
ENV = {"HF_HOME": f"{CACHE}/hf"}

app = modal.App("laya-finetune")


def base_dir() -> str:
    """The base checkpoint's folder, downloaded into the volume on first use."""
    from huggingface_hub import snapshot_download

    sub = CHECKPOINTS[CHECKPOINT]
    root = snapshot_download(
        REPO,
        allow_patterns=[
            f"{sub}/{name}"
            for name in (
                "rl_agent_config.json",
                "model.safetensors",
                "tokenizer/*",
                "encoder/*",
            )
        ],
    )
    return os.path.join(root, sub)


@app.function(image=image, env=ENV, volumes={CACHE: volume}, timeout=30 * 60)
def fetch_base():
    """Download once, before the folds start, so no two containers write the cache."""
    from laya.agent import _fix_tokenizer_config

    _fix_tokenizer_config(base_dir())
    volume.commit()


# train_ddp.py, unchanged


def collate_train_batch(items, pad_id):
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def fit_one_temp(sel):
    import torch

    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@app.function(
    image=image, gpu=GPU, env=ENV, volumes={CACHE: volume}, timeout=2 * 60 * 60
)
def finetune(fold: int, split: dict, variant: dict, epochs: int = EPOCHS) -> list[dict]:
    import laya
    import torch
    from laya.common import (
        QTYPES,
        build_model,
        build_sequence,
        proper_reward,
        serialize_state,
    )
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    random.seed(SEED + fold)
    torch.manual_seed(SEED + fold)
    device = torch.device("cuda")

    with open(f"{DATASET_DIR}/{KIND}.csv", newline="") as f:
        by_id = {s["id"]: s for s in csv.DictReader(f)}
    train, calib, test = (
        [by_id[i] for i in split[role]] for role in ("train", "calib", "test")
    )

    base = base_dir()
    with open(os.path.join(base, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(base, "tokenizer"))

    q = variant["question"]
    b = budgets(tok, cfg, q)
    internal = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
    # option index -> expert
    keys = [variant["labels"][label] for label in q["criteria"]]
    k = len(keys)

    def item(sample, order):
        """`order[j]` is the option at position j; the target follows the gold option."""
        ids, markers = build_sequence(
            tok, json.loads(sample["turns"]), internal, b["max_len"], b["head_max_len"],
            option_order=order, truncate_left=True,
        )  # fmt: skip
        assert len(markers) == k, sample["id"]
        target = [0.0] * k
        target[order.index(keys.index(sample["gold"]))] = 1.0
        return {
            "ids": ids,
            "markers": markers,
            "qtype": QTYPES["choice"],
            "target": target,
        }

    model = build_model(cfg, encoder_dir=os.path.join(base, "encoder"))
    model.load_state_dict(
        load_file(os.path.join(base, "model.safetensors")), strict=True
    )
    model.encoder.config.reference_compile = False  # as laya.Agent: no torch.compile
    model.to(device).train()

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": LR_ENCODER},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=0.01,
    )
    n_micro = -(-len(train) // MICRO_BATCH)
    total_updates = -(-n_micro // GRAD_ACCUM) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6
    )
    print(
        f"[ft] fold {fold}: {len(train)} train, {len(calib)} calib, {len(test)} test | "
        f"max_len={b['max_len']} head_max_len={b['head_max_len']} | {epochs} epochs"
    )

    for epoch in range(epochs):
        # a fresh option order per sample and epoch
        items = [item(s, random.sample(range(k), k)) for s in train]
        random.shuffle(items)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, epochs - 1)
        epoch_loss = epoch_reward = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, start in enumerate(range(0, len(items), MICRO_BATCH)):
            batch = collate_train_batch(
                items[start : start + MICRO_BATCH], tok.pad_token_id
            )
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            n_opts = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)

            # 1. GROUP_SIZE noisy logit vectors, the noise projected to zero mean
            eps = (
                torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            )
            eps = (eps - eps.sum(-1, keepdim=True) / n_opts) * mask
            z = logits.detach().unsqueeze(0) + eps
            probs = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. proper scoring reward, advantage against the group mean
            with torch.no_grad():
                r = proper_reward(
                    probs, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0
                )
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. policy gradient (Gaussian log-likelihood of the noise) + soft cross-entropy
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = (
                -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1))
                .sum(-1)
                .mean()
            )
            loss = (loss_rl + loss_ce) / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or step + 1 == n_micro:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * GRAD_ACCUM
            epoch_reward += r.mean().item()
        print(
            f"[ft] fold {fold} epoch {epoch + 1}/{epochs}: loss={epoch_loss / n_micro:.4f} "
            f"reward={epoch_reward / n_micro:.3f} sigma={sigma:.2f}"
        )

    # temperature on the calibration slice, options in the order inference uses
    model.eval()
    del optimizer, scheduler
    torch.cuda.empty_cache()
    canonical = list(range(k))
    sel = []
    with torch.no_grad():
        for start in range(0, len(calib), 16):
            chunk = [item(s, canonical) for s in calib[start : start + 16]]
            cb = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits, _ = model(
                    cb["input_ids"].to(device),
                    cb["attention_mask"].to(device),
                    cb["marker_pos"].to(device),
                    cb["marker_mask"].to(device),
                    cb["qtype"].to(device),
                )
            sel += [
                (z[:k], it["target"])
                for z, it in zip(logits.float().cpu().numpy(), chunk)
            ]
    t_fit = fit_one_temp(sel)
    print(f"[ft] fold {fold}: fitted T={t_fit:.3f} on {len(sel)} calib samples")

    out = f"{CACHE}/ft/{KIND}/fold{fold}"
    os.makedirs(out, exist_ok=True)
    save_file(
        {n: v.half().contiguous().cpu() for n, v in model.state_dict().items()},
        os.path.join(out, "model.safetensors"),
    )
    model.encoder.config.save_pretrained(os.path.join(out, "encoder"))
    tok.save_pretrained(os.path.join(out, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = f"{MODEL}-{KIND}-fold{fold}"
    cfg["temperature"] = [t_fit, *cfg.get("temperature", [1.0, 1.0, 1.0])[1:]]
    cfg.pop("temperature_by_options", None)  # a bucket would override the fitted T
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    volume.commit()
    del model
    torch.cuda.empty_cache()

    # the held-out fold, through laya's own inference path
    agent = laya.load(out, device=device.type)
    states = [json.loads(s["turns"]) for s in test]
    results = agent.predict_batch(
        states,
        {"expert": q},
        batch_size=BATCH_SIZE,
        max_len=b["max_len"],
        head_max_len=b["head_max_len"],
        sort_by_length=True,
    )
    name = "|".join(VARIANT)
    rows = []
    for sample, state, res in zip(test, states, results):
        n_state = len(
            tok(serialize_state(state), add_special_tokens=False)["input_ids"]
        )
        row = result_row(
            sample, MODEL, name, variant, res["answers"]["expert"], n_state, b
        )
        # laya clamps T to [0.5, 5] at load; this is the one it applied
        rows.append({**row, "temperature": agent.temperature[QTYPES["choice"]]})
    print(f"[ft] fold {fold}: acc={sum(r['correct'] for r in rows) / len(rows):.3f}")
    return rows


@app.local_entrypoint()
def main(
    folds: str = ",".join(map(str, range(N_FOLDS))),
    epochs: int = EPOCHS,
    out_dir: str = "data/sim/laya/finetune",
):
    import numpy as np
    from sklearn.model_selection import GroupKFold

    with open(f"dataset/{KIND}.csv", newline="") as f:
        samples = list(csv.DictReader(f))
    ids = np.array([s["id"] for s in samples])
    groups = np.array([s["scenario_id"] for s in samples])
    y = np.array([s["gold"] for s in samples])

    wanted = {int(k) for k in folds.split(",")}
    splits = {}
    # analysis.probe.folds_for: the same call, so the same folds
    for k, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, groups)
    ):
        if k not in wanted:
            continue
        scenarios = sorted(set(groups[train_idx]))
        n_calib = max(1, round(CALIB_SHARE * len(scenarios)))
        held = set(random.Random(SEED + k).sample(scenarios, n_calib))
        splits[k] = {
            "train": [str(ids[i]) for i in train_idx if groups[i] not in held],
            "calib": [str(ids[i]) for i in train_idx if groups[i] in held],
            "test": [str(i) for i in ids[test_idx]],
        }

    variant = build_variants(*([v] for v in VARIANT))["|".join(VARIANT)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps(splits, indent=2))

    fetch_base.remote()
    jobs = [(k, split, variant, epochs) for k, split in splits.items()]
    rows = [r for batch in finetune.starmap(jobs) for r in batch]

    path = out / f"{KIND}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"[ft] wrote {len(rows)} rows -> {path} (acc={acc:.3f})")
