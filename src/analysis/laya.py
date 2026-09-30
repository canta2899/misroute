"""Laya, a non-generative decision model, against the coordinators, their probes and the
text baselines.

Every checkpoint is run on a grid of prompt variants (`options`, `instructions`,
`state`; see `simulate/laya_app.py`). Laya is not trained on our labels, but choosing the
best variant is: `select` picks it per held-out fold on the other folds, so the accuracy
of the chosen variant is out of fold, like every other router's.

`laya-multilingual-ft` is `multilingual` fine-tuned with RLCD on the other folds, with a
temperature fitted on a held-out slice of them (`simulate/laya_finetune_app.py`). It runs
one variant, so its rows are already one per sample and out of fold. `select_all` joins
it to the selected zero-shot rows. The comparisons below all read that frame: one row per
(checkpoint, sample).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from analysis import probe as P
from analysis.paths import LAYA, LAYA_FT

FAMILIES = ("laya", "baseline", "coordinator", "probe")
VARIANT_FACTORS = ("options", "instructions", "state")


def kinds() -> list[str]:
    """Every kind with a laya csv on disk."""
    return [k for k in P.KINDS if (LAYA / f"{k}.csv").exists()]


def load(kind: str) -> pd.DataFrame:
    """One row per (checkpoint, variant, sample): the laya csv, the design factors and the fold."""
    return _read(LAYA / f"{kind}.csv", kind)


def load_finetuned(kind: str) -> pd.DataFrame | None:
    """The fine-tuned checkpoint, one row per sample, with the fold's `temperature`;
    None when that kind was not fine-tuned."""
    path = LAYA_FT / f"{kind}.csv"
    return _read(path, kind) if path.exists() else None


def select_all(t: pd.DataFrame, ft: pd.DataFrame | None) -> pd.DataFrame:
    """The zero-shot checkpoints with their chosen variants, plus the fine-tuned one."""
    return (
        pd.concat([select(t), ft], ignore_index=True) if ft is not None else select(t)
    )


def _read(path, kind: str) -> pd.DataFrame:
    t = pd.read_csv(path).rename(columns={"model": "checkpoint"})
    ds = P.dataset(kind)
    ds["fold"] = P.folds_for(ds["scenario_id"].to_numpy(), ds["gold"].to_numpy())
    factors = [f for f in P.FACTORS if f != "gold"]
    return t.merge(
        ds[["id", "fold", "scenario_id", *factors]],
        on="id",
        how="left",
        validate="many_to_one",
    )


def checkpoints(t: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(t["checkpoint"]))


def _slices(construction):
    return [("ALL", np.ones(len(construction), bool))] + [
        (c, construction == c) for c in P.CONSTRUCTIONS if (construction == c).any()
    ]


# variants


def variant_table(t: pd.DataFrame) -> pd.DataFrame:
    """Per (checkpoint, variant): accuracy, the expert predicted most and its share."""
    g = t.groupby(["checkpoint", "variant"], sort=False)
    shares = g["predicted"].value_counts(normalize=True)
    first = shares.groupby(level=["checkpoint", "variant"], sort=False).head(1)
    out = g[list(VARIANT_FACTORS)].first()
    out["acc"] = g["correct"].mean()
    out["top_expert"] = first.reset_index(level="predicted")["predicted"]
    out["top_share"] = first.droplevel("predicted")
    out["options_cut"] = g["options_cut"].any()
    return out


def variant_grid(t: pd.DataFrame, value: str = "acc") -> pd.DataFrame:
    """`variant_table[value]` as variant (rows) x checkpoint (columns), best mean accuracy first."""
    v = variant_table(t)
    order = v["acc"].groupby(level="variant").mean().sort_values(ascending=False).index
    return v[value].unstack("checkpoint").loc[order, checkpoints(t)]


def factor_effects(t: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per level of each variant factor, averaged over the other two, per checkpoint."""
    v = variant_table(t).reset_index()
    return pd.concat(
        {
            f: v.pivot_table(index=f, columns="checkpoint", values="acc", sort=False)
            for f in VARIANT_FACTORS
        },
        names=["factor", "level"],
    )[checkpoints(t)]


def select(t: pd.DataFrame) -> pd.DataFrame:
    """One row per (checkpoint, sample): the variant chosen on the other four folds.

    For each held-out fold, the variant with the highest accuracy on the remaining folds
    answers that fold, so no sample is scored by a variant chosen on it.
    """
    out = []
    for _, g in t.groupby("checkpoint", sort=False):
        for f in range(P.N_FOLDS):
            train = g[g["fold"] != f].groupby("variant", sort=False)["correct"].mean()
            out.append(g[(g["fold"] == f) & (g["variant"] == train.idxmax())])
    return pd.concat(out, ignore_index=True)


def selection(s: pd.DataFrame) -> pd.DataFrame:
    """Which variant each checkpoint chose for each held-out fold."""
    return (
        s.groupby(["checkpoint", "fold"], sort=False)["variant"]
        .first()
        .unstack("fold")
        .rename(columns=lambda f: f"fold {f}")
    )


def finetune_folds(ft: pd.DataFrame) -> pd.DataFrame:
    """Per held-out fold of the fine-tuned checkpoint: accuracy, mean confidence and the
    temperature fitted on its calibration slice."""
    return ft.groupby("fold").agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        confidence=("answer_confidence", "mean"),
        temperature=("temperature", "first"),
    )


# accuracy


def fold_table(s: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Per-fold accuracy of every router, one row per (family, router, slice).

    `s` is the selected laya frame (`select`); `results` are the probe runs of the same
    kind (`probe.result`). The coordinator and the probe give one row per run, laya one
    per checkpoint, the baselines one each.
    """
    kind = s["kind"].iloc[0]
    rows = []
    for ckpt, g in s.groupby("checkpoint", sort=False):
        correct, folds = g["correct"].to_numpy(bool), g["fold"].to_numpy()
        for sl, m in _slices(g["construction"].to_numpy()):
            rows.append(("laya", ckpt, sl, P.fold_accuracy(correct, folds, m)))
    per_fold = P.baselines(kind)["per_fold"]
    for name in P.BASELINE_NAMES:
        for sl, folds in per_fold[name].items():
            rows.append(("baseline", name, sl, np.asarray(folds, float)))
    for d in results.values():
        f = P.compare_folds(d)
        for sl in f.index:
            rows.append(("coordinator", d["model"], sl, f.loc[sl, "model"]))
            rows.append(("probe", d["model"], sl, f.loc[sl, "probe"]))
    return pd.DataFrame(rows, columns=["family", "router", "slice", "per_fold"])


def compare(s: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Fold-averaged accuracy with the worst and best fold, long form."""
    t = fold_table(s, results)
    t["acc"] = t["per_fold"].map(np.mean)
    t["fold_min"] = t["per_fold"].map(np.min)
    t["fold_max"] = t["per_fold"].map(np.max)
    return t.drop(columns="per_fold")


def compare_wide(s: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Accuracy per router (rows) and slice (columns)."""
    t = compare(s, results)
    w = t.pivot_table(
        index=["family", "router"], columns="slice", values="acc", sort=False
    )
    w = w[[sl for sl in ("ALL", *P.CONSTRUCTIONS) if sl in w]]
    return w.reindex(sorted(w.index, key=lambda i: FAMILIES.index(i[0])))


def summary(s: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Laya and the baselines one by one; the coordinators and the probes as mean, min,
    max over runs."""
    w = compare_wide(s, results)
    single = w.loc[["laya", "baseline"]]
    single.index = [r for _, r in single.index]
    runs = {}
    for family in ("coordinator", "probe"):
        g = w.loc[family]
        for stat in ("mean", "min", "max"):
            runs[f"{family} ({stat})"] = getattr(g, stat)()
    return pd.concat([single, pd.DataFrame(runs).T])


# what laya predicts


def prediction_shares(s: pd.DataFrame) -> pd.DataFrame:
    """Share of the samples routed to each expert, per checkpoint, beside the gold shares."""
    shares = pd.crosstab(s["checkpoint"], s["predicted"], normalize="index")
    first = s[s["checkpoint"] == checkpoints(s)[0]]
    shares.loc["gold"] = first["gold"].value_counts(normalize=True)
    return shares.reindex(columns=list(P.EXPERTS), fill_value=0.0)


def pair_outcomes(s: pd.DataFrame) -> pd.DataFrame:
    """Adversarial pairs per checkpoint: `both`, `one` or `neither` sample right, and
    `collapsed` (same expert for both), as in `probe.pair_outcomes`."""
    s = s[s["construction"] == "adversarial"]
    rows = {}
    for ckpt, g in s.groupby("checkpoint", sort=False):
        sizes = g.groupby("request_id").size()
        g = g[g["request_id"].isin(sizes[sizes == 2].index)]
        by_pair = g.groupby("request_id")
        right = by_pair["correct"].sum()
        rows[ckpt] = {
            "pairs": len(right),
            "both": (right == 2).mean(),
            "one": (right == 1).mean(),
            "neither": (right == 0).mean(),
            "collapsed": (by_pair["predicted"].nunique() == 1).mean(),
        }
    return pd.DataFrame(rows).T


def pair_compare(s: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Laya's pair outcomes beside the other routers (mean over runs)."""
    cols = ["pairs", "both", "one", "neither", "collapsed"]
    kind = s["kind"].iloc[0]
    others = (
        P.pair_summary(results).loc[kind, cols].rename(index={"model": "coordinator"})
    )
    return pd.concat([pair_outcomes(s)[cols].astype(float), others])


def error_direction(s: pd.DataFrame) -> pd.DataFrame:
    """Per checkpoint and tier: share wrong (`error`), wrong on the decoy (`decoy`), wrong
    on another expert (`other`), as in `probe.error_direction`."""
    rows = {}
    for ckpt, g in s.groupby("checkpoint", sort=False):
        for tier in P.DEPENDENCE_TIERS:
            m = g[g["construction"] == tier]
            wrong = ~m["correct"].to_numpy(bool)
            on_decoy = (m["predicted"] == m["decoy"]).to_numpy()
            rows[(ckpt, tier)] = {
                "n": len(m),
                "error": wrong.mean(),
                "decoy": on_decoy.mean(),
                "other": (wrong & ~on_decoy).mean(),
                "decoy_share": on_decoy[wrong].mean(),
            }
    return pd.DataFrame(rows).T.rename_axis(["router", "tier"])


def error_direction_compare(s: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Laya's error direction beside the other routers (mean over runs)."""
    cols = ["error", "decoy", "other", "decoy_share"]
    kind = s["kind"].iloc[0]
    others = P.error_direction_summary(results).loc[kind, cols]
    others = others.rename(index={"model": "coordinator"}, level="router")
    return pd.concat([error_direction(s)[cols].astype(float), others])


# sample by sample


def dependence(s: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Does laya get right the samples a coordinator, or its probe, gets wrong?

    One row per (checkpoint, run, router): laya's accuracy where that router is right and
    where it is wrong, the `gap` between the two, and Cohen's kappa on right/wrong.
    """
    rows = {}
    for d in results.values():
        c = P.cells(d).set_index("id")
        for ckpt, g in s.groupby("checkpoint", sort=False):
            j = c.join(g.set_index("id")["correct"].rename("laya_ok"), how="inner")
            ok = j["laya_ok"].to_numpy(bool)
            for router, name in (("model", "coordinator"), ("probe", "probe")):
                other = j[f"{router}_ok"].to_numpy(bool)
                rows[(ckpt, d["model"], name)] = {
                    "n": len(j),
                    "acc_router_right": ok[other].mean(),
                    "acc_router_wrong": ok[~other].mean(),
                    "gap": ok[~other].mean() - ok[other].mean(),
                    "laya_only": (ok & ~other).mean(),
                    "kappa": cohen_kappa_score(ok, other),
                }
    return (
        pd.DataFrame(rows).T.rename_axis(["checkpoint", "run", "router"]).astype(float)
    )


def dependence_summary(s: pd.DataFrame, results: dict) -> pd.DataFrame:
    """`dependence` averaged over runs, with the range of `gap`."""
    t = dependence(s, results).reset_index()
    g = t.groupby(["checkpoint", "router"], sort=False)
    out = g[
        ["acc_router_right", "acc_router_wrong", "gap", "laya_only", "kappa"]
    ].mean()
    out["gap_min"], out["gap_max"] = g["gap"].min(), g["gap"].max()
    out["n_runs"] = g.size()
    return out


# confidence


def reliability(s: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Accuracy per bin of `answer_confidence`, per checkpoint (empty bins left out)."""
    edges = np.linspace(0, 1, bins + 1)
    s = s.assign(bin=pd.cut(s["answer_confidence"], edges, include_lowest=True))
    r = s.groupby(["checkpoint", "bin"], observed=True, sort=False).agg(
        confidence=("answer_confidence", "mean"),
        acc=("correct", "mean"),
        n=("correct", "size"),
    )
    return r.reset_index().sort_values(["checkpoint", "bin"], ignore_index=True)


def calibration(s: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Per checkpoint: mean confidence against accuracy, the expected calibration error
    (`ece`) and how well confidence separates right from wrong (`auroc`, 0.5 = not at all)."""
    r = reliability(s, bins)
    rows = {}
    for ckpt, g in s.groupby("checkpoint", sort=False):
        rb = r[r["checkpoint"] == ckpt]
        correct = g["correct"].to_numpy(bool)
        rows[ckpt] = {
            "acc": correct.mean(),
            "confidence": g["answer_confidence"].mean(),
            "ece": float(
                (rb["n"] * (rb["confidence"] - rb["acc"]).abs()).sum() / rb["n"].sum()
            ),
            "auroc": roc_auc_score(correct, g["answer_confidence"])
            if 0 < correct.sum() < len(correct)
            else np.nan,
        }
    return pd.DataFrame(rows).T
