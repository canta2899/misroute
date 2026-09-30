"""Frozen ModernBERT-large as an embedding model + logistic regression, out of fold.

    uv run modal run -m simulate.modernbert_frozen_app
    uv run modal run -m simulate.modernbert_frozen_app --kinds single_turn

No fine-tuning: the encoder only reads the text. Each sample's vector is the mean of
the last layer over its tokens; a logistic regression (StandardScaler +
LogisticRegression(C=1, max_iter=2000), as the other baselines) trains on it, out of
fold on the same scenario-grouped `GroupKFold(5)` as the probes and the baselines.
One container per kind.

Input: the `user_text` column, the same text as the three dataset baselines
(`analysis.probe.baselines`).

Output (local): <out-dir>/<kind>.csv, one row per sample with the fold, the
prediction and p_<expert> for every expert.
"""

import csv
from pathlib import Path

import modal

from simulate.modernbert_app import (
    BASE,
    CACHE,
    DATASET_DIR,
    ENV,
    EXPERTS,
    GPU,
    KINDS,
    MAX_LEN,
    N_FOLDS,
    volume,
)
from simulate.modernbert_app import image as ft_image

MODEL = "modernbert-large-frozen"
BATCH = 32
C = 1.0  # analysis.probe.C
SEED = 0

image = ft_image.add_local_python_source("simulate")
app = modal.App("modernbert-frozen")


@app.function(image=image, gpu=GPU, env=ENV, volumes={CACHE: volume}, timeout=60 * 60)
def run(kind: str, folds: list) -> list[dict]:
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    with open(f"{DATASET_DIR}/{kind}.csv", newline="") as f:
        samples = list(csv.DictReader(f))

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.truncation_side = "left"
    model = AutoModel.from_pretrained(BASE).to("cuda").eval()

    vectors = []
    with torch.no_grad():
        for start in range(0, len(samples), BATCH):
            batch = tok(
                [s["user_text"] for s in samples[start : start + BATCH]],
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            H = model(**batch).last_hidden_state.float()
            m = batch["attention_mask"][:, :, None].float()
            vectors.append(((H * m).sum(1) / m.sum(1)).cpu().numpy())
    X = np.concatenate(vectors)
    print(f"[frozen] {kind}: embedded {X.shape}")

    y = np.array([s["gold"] for s in samples])
    folds = np.array(folds)
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(EXPERTS)))
    for f in range(N_FOLDS):
        held = folds == f
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=C, random_state=SEED)
        ).fit(X[~held], y[~held])
        order = [list(clf.classes_).index(e) for e in EXPERTS]
        pred[held] = clf.predict(X[held])
        proba[held] = clf.predict_proba(X[held])[:, order]
        print(f"[frozen] {kind} fold {f}: acc={(pred[held] == y[held]).mean():.3f}")

    return [
        {
            "id": s["id"],
            "kind": kind,
            "model": MODEL,
            "fold": int(fold),
            "construction": s["construction"],
            "gold": s["gold"],
            "predicted": p,
            "correct": p == s["gold"],
            **{f"p_{e}": float(v) for e, v in zip(EXPERTS, pr)},
        }
        for s, fold, p, pr in zip(samples, folds, pred, proba)
    ]


@app.local_entrypoint()
def main(kinds: str = ",".join(KINDS), out_dir: str = "data/sim/modernbert/frozen"):
    import numpy as np
    from sklearn.model_selection import GroupKFold

    kinds = kinds.split(",")
    jobs = []
    for kind in kinds:
        with open(f"dataset/{kind}.csv", newline="") as f:
            samples = list(csv.DictReader(f))
        y = np.array([s["gold"] for s in samples])
        groups = np.array([s["scenario_id"] for s in samples])
        folds = np.empty(len(y), int)
        # analysis.probe.folds_for: the same call, so the same folds
        for k, (_, held) in enumerate(
            GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, groups)
        ):
            folds[held] = k
        jobs.append((kind, folds.tolist()))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for (kind, _), rows in zip(jobs, list(run.starmap(jobs))):
        path = out / f"{kind}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"[frozen] wrote {len(rows)} rows -> {path} (pooled acc={acc:.3f})")
