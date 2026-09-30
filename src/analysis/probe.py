from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.paths import BASELINE_CACHE, CACHE, DATA, DATASET_DIR

C = 1.0  # logreg inverse regularization
N_FOLDS = 5
SEED = 0
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # same encoder as the dataset baselines


KINDS = ("single_turn", "multi_turn")
CONSTRUCTIONS = ("plain", "natural", "adversarial")
EXPERTS = (
    "time_to_market",
    "scale_arch",
    "resource_opt",
    "legacy_compat",
    "strict_compliance",
)
CHANCE = 1 / len(EXPERTS)
DECIMALS = 3  # the precision every table and figure of the thesis prints


def shown(x):
    """A value as the thesis prints it. Every difference, mean and ratio a table or a figure
    reports is taken on these, so subtracting two printed numbers gives the printed one."""
    return np.round(x, DECIMALS)


def runs() -> list[tuple[str, str]]:
    """Every (model, kind) with a parquet on disk, models in size order."""
    order = [
        "gemma4-e2b-it",
        "qwen3-4b",
        "qwen3-8b",
        "gpt-oss-20b",
        "gemma4-12b-it",
        "gemma4-26b-a4b-it",
        "llama3.3-70b",
        "qwen2.5-72b",
    ]
    found = {
        p.name.rsplit("_", 2)[0] + "|" + "_".join(p.stem.split("_")[-2:])
        for p in DATA.glob("*.parquet")
    }
    found = {(m, k) for m, k in (f.split("|") for f in found)}
    return [(m, k) for m in order for k in KINDS if (m, k) in found] + sorted(
        r for r in found if r[0] not in order
    )


# folds


def folds_for(groups, y):
    """Scenario-grouped folds: a contrastive pair shares one, so splitting by sample
    cannot put one member in train and its twin in test."""
    folds = np.empty(len(y), int)
    for i, (_, held_out) in enumerate(
        GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, groups)
    ):
        folds[held_out] = i
    return folds


def logreg():
    return LogisticRegression(max_iter=2000, C=C, random_state=SEED)


def out_of_fold(folds, y, fit_predict):
    """One prediction per sample, always from a model that never saw its scenario."""
    pred = np.empty(len(y), dtype=object)
    for f in range(N_FOLDS):
        held_out = folds == f
        pred[held_out] = fit_predict(~held_out, held_out)
    return pred


def fold_accuracy(correct, folds, mask=None):
    """Accuracy inside each held-out fold - the spread the plots draw as error bars."""
    mask = np.ones(len(folds), bool) if mask is None else mask
    return np.array([correct[mask & (folds == f)].mean() for f in range(N_FOLDS)])


# baselines

_encoder = None


def embed(texts):
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(EMBEDDING_MODEL)
        # a few multi-turn transcripts exceed the 512-token window; the deciding turn is
        # the last one, so drop from the front
        _encoder.tokenizer.truncation_side = "left"
    return np.asarray(
        _encoder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    )


def dataset(kind: str) -> pd.DataFrame:
    """The exported csv for one kind, string columns as numpy object."""
    df = pd.read_csv(DATASET_DIR / f"{kind}.csv")
    for col in df.columns:
        if df[col].dtype == "str":
            df[col] = df[col].astype(object)
    for col in ("decoy", "pattern", "variant", "pivot", "collision"):
        if col in df:
            df[col] = df[col].fillna("none")
    return df


BASELINE_NAMES = ("lexical", "tfidf", "embedding")
ROUTERS = ("model", "lexical", "tfidf", "embedding", "probe")


def baselines(kind: str) -> dict:
    """Lexical (bag of words), tf-idf and embedding accuracy for one kind, cached on disk."""
    cache = json.loads(BASELINE_CACHE.read_text()) if BASELINE_CACHE.exists() else {}
    df = dataset(kind)
    fingerprint = hashlib.sha1("".join(sorted(df["id"])).encode()).hexdigest()[:16]
    cached = cache.get(kind, {})
    if (
        cached.get("fingerprint") == fingerprint
        and set(BASELINE_NAMES) <= set(cached.get("accuracy", {}))
        and set(BASELINE_NAMES) <= set(cached.get("per_fold", {}))
        and set(BASELINE_NAMES) <= set(cached.get("pred", {}))
    ):
        return cached

    text = df["user_text"].to_numpy()
    y = df["gold"].to_numpy()
    construction = df["construction"].to_numpy()
    folds = folds_for(df["scenario_id"].to_numpy(), y)

    def bow(train, test):
        clf = make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=2), logreg())
        clf.fit(text[train], y[train])
        return clf.predict(text[test])

    def tfidf(train, test):
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), logreg())
        clf.fit(text[train], y[train])
        return clf.predict(text[test])

    X = embed(text)
    entry = {
        "fingerprint": fingerprint,
        "accuracy": {},
        "per_fold": {},
        "pred": {},
        "ids": df["id"].tolist(),
    }
    for name, fit_predict in [
        ("lexical", bow),
        ("tfidf", tfidf),
        ("embedding", lambda tr, te: logreg().fit(X[tr], y[tr]).predict(X[te])),
    ]:
        pred = out_of_fold(folds, y, fit_predict)
        correct = pred == y
        slices = [("ALL", np.ones(len(y), bool))] + [
            (c, construction == c) for c in CONSTRUCTIONS if (construction == c).any()
        ]
        entry["accuracy"][name] = {s: float(correct[m].mean()) for s, m in slices}
        entry["per_fold"][name] = {
            s: fold_accuracy(correct, folds, m).tolist() for s, m in slices
        }
        entry["pred"][name] = pred.tolist()
    cache[kind] = entry
    BASELINE_CACHE.write_text(json.dumps(cache, indent=2))
    return entry


# probes


# where a hidden state was read: the last prompt token (every model) or the last token of
# the reasoning block (gpt-oss-20b only, whose reasoning cannot be turned off)
SITES = {"prompt": "hidden_states", "reasoning": "reasoning_hidden_states"}


def reasoning_runs() -> list[tuple[str, str]]:
    """Every (model, kind) whose parquet also carries hidden states at the last reasoning token."""
    import pyarrow.parquet as pq

    return [
        (m, k)
        for m, k in runs()
        if f"{m}_{SITES['reasoning']}"
        in pq.read_schema(DATA / f"{m}_{k}.parquet").names
    ]


def load(model: str, kind: str, site: str = "prompt") -> dict:
    """
    One run: the sample table plus the hidden states at the last prompt token
    (or at the last reasoning token, with `site="reasoning"`).
    """
    df = pd.read_parquet(DATA / f"{model}_{kind}.parquet")
    col = lambda s: df[f"{model}_{s}"]  # noqa: E731
    n_layers, hidden_dim = int(col("n_layers").iloc[0]), int(col("hidden_dim").iloc[0])
    dtype = col("hidden_dtype").iloc[0]
    H = np.stack(
        [
            np.frombuffer(b, dtype=dtype).reshape(n_layers, hidden_dim)
            for b in col(SITES[site])
        ]
    ).astype(np.float32)

    predicted = col("predicted").to_numpy()
    n_unrouted = int(pd.isna(predicted).sum())
    y = df["gold"].to_numpy()
    return {
        "model": model,
        "kind": kind,
        "site": site,
        "label": f"{model} {kind}",
        "df": df,
        "H": H,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_unrouted": n_unrouted,
        "y": y,
        "predicted": predicted,
        "construction": df["construction"].to_numpy(),
        "folds": folds_for(df["scenario_id"].to_numpy(), y),
        "experts": list(EXPERTS),
    }


def _fit(d, layer, train):
    X = d["H"][train, layer, :]
    return make_pipeline(StandardScaler(), logreg()).fit(X, d["y"][train])


def probe_predictions(d, layer):
    H = d["H"]
    return out_of_fold(
        d["folds"], d["y"], lambda tr, te: _fit(d, layer, tr).predict(H[te, layer, :])
    )


def probe_probabilities(d, layer):
    proba = np.zeros((len(d["y"]), len(d["experts"])))
    for f in range(N_FOLDS):
        held_out = d["folds"] == f
        clf = _fit(d, layer, ~held_out)
        order = [list(clf.classes_).index(e) for e in d["experts"]]
        proba[held_out] = clf.predict_proba(d["H"][held_out, layer, :])[:, order]
    return proba


# sweep cache


def cache_path(model, kind, site="prompt"):
    suffix = "" if site == "prompt" else f"_{site}"
    return CACHE / f"{model}_{kind}{suffix}.npz"


def sweep(model: str, kind: str, verbose: bool = True, site: str = "prompt") -> dict:
    """Out-of-fold probe predictions for every layer, computed once and cached."""
    path = cache_path(model, kind, site)
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if not bool(z["partial"]):
            return {k: z[k] for k in z.files}

    d = load(model, kind, site)
    n_layers, n = d["n_layers"], len(d["y"])
    oof = np.full((n_layers, n), "", dtype=object)
    done = 0
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if len(z["oof"]) == n_layers and len(z["oof"][0]) == n:
            oof, done = z["oof"], int(z["done"])

    CACHE.mkdir(parents=True, exist_ok=True)
    for layer in range(done, n_layers):
        oof[layer] = probe_predictions(d, layer)
        if verbose:
            print(
                f"{model} {kind} {site} layer {layer + 1}/{n_layers} acc={(oof[layer] == d['y']).mean():.3f}",
                flush=True,
            )
        _save(path, d, oof, layer + 1, partial=layer + 1 < n_layers)

    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _save(path, d, oof, done, partial, proba=None, peak=None):
    payload = dict(
        oof=oof,
        done=done,
        partial=partial,
        y=d["y"],
        predicted=d["predicted"],
        construction=d["construction"],
        folds=d["folds"],
        ids=d["df"]["id"].to_numpy(),
        experts=np.array(d["experts"]),
        n_layers=d["n_layers"],
        hidden_dim=d["hidden_dim"],
        n_unrouted=d["n_unrouted"],
    )
    if proba is not None:
        payload |= dict(proba=proba, proba_layer=peak)
    elif path.exists():
        old = np.load(path, allow_pickle=True)
        if "proba" in old.files:
            payload |= dict(proba=old["proba"], proba_layer=old["proba_layer"])
    np.savez(path, **payload)


def probabilities(
    model: str, kind: str, layer: int | None = None, site: str = "prompt"
) -> tuple[np.ndarray, int]:
    z = sweep(model, kind, site=site)
    layer = peak_layer(z) if layer is None else layer
    if "proba" in z and int(z["proba_layer"]) == layer:
        return z["proba"], layer
    d = load(model, kind, site)
    proba = probe_probabilities(d, layer)
    _save(
        cache_path(model, kind, site),
        d,
        z["oof"],
        int(z["done"]),
        False,
        proba=proba,
        peak=layer,
    )
    return proba, layer


def auc_macro(d, proba) -> float:
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    Y = label_binarize(d["y"], classes=d["experts"])
    grid = np.linspace(0, 1, 200)
    tpr = [
        np.interp(grid, *roc_curve(Y[:, i], proba[:, i])[:2])
        for i in range(len(d["experts"]))
    ]
    return float(auc(grid, np.mean(tpr, axis=0)))


# tables


def result(model: str, kind: str, site: str = "prompt") -> dict:
    """A cached sweep as the dict the plots and tables read."""
    z = sweep(model, kind, site=site)
    y, oof = z["y"], z["oof"]
    correct = oof == y
    folds = z["folds"]
    construction = z["construction"]
    constructions = [c for c in CONSTRUCTIONS if (construction == c).any()]

    def fold_avg(mask):
        """Per-layer accuracy, averaged over the 5 held-out folds - never pooled."""
        return np.stack(
            [correct[:, mask & (folds == f)].mean(1) for f in range(N_FOLDS)], axis=1
        ).mean(1)

    return {
        "model": model,
        "kind": kind,
        "site": site,
        "label": f"{model} {kind}",
        "y": y,
        "oof": oof,
        "correct": correct,
        "predicted": z["predicted"],
        "construction": construction,
        "constructions": constructions,
        "folds": z["folds"],
        "experts": list(z["experts"]),
        "n_layers": int(z["n_layers"]),
        "hidden_dim": int(z["hidden_dim"]),
        "n_unrouted": int(z["n_unrouted"]),
        "ids": z["ids"],
        "baselines": baselines(kind),
        "by_layer": pd.DataFrame(
            {"acc": fold_avg(np.ones(len(y), bool))}
            | {c: fold_avg(construction == c) for c in constructions},
            index=pd.RangeIndex(int(z["n_layers"]), name="layer"),
        ),
    }


def split_label(label: str) -> tuple[str, str]:
    model, kind = label.rsplit(" ", 1)
    return model, kind


def overview(results=None) -> pd.DataFrame:
    results = results or {f"{m} {k}": result(m, k) for m, k in runs()}
    t = shown(across_runs(compare, results=results)[list(ROUTERS)].astype(float))
    t["margin"] = t["probe"] - t["model"]
    t["kind"] = [split_label(i)[1] for i in t.index]
    t["model_name"] = [split_label(i)[0] for i in t.index]
    return t


def tiers(d) -> pd.DataFrame:
    t = shown(compare(d)[["model", "probe"]].astype(float))
    t["margin"] = t["probe"] - t["model"]
    return t


def tier_table(results) -> pd.DataFrame:
    t = pd.concat(
        {label: tiers(d) for label, d in results.items()}, names=["run", "tier"]
    )
    t["kind"] = [split_label(r)[1] for r in t.index.get_level_values("run")]
    t["model_name"] = [split_label(r)[0] for r in t.index.get_level_values("run")]
    return t


def tier_drop(results) -> pd.DataFrame:
    """Per run, how much accuracy the model and the probe lose from plain to adversarial."""
    t = tier_table(results)
    rows = {}
    for run, g in t.groupby(level="run", sort=False):
        g = g.droplevel("run")
        drop_model = g.loc["plain", "model"] - g.loc["adversarial", "model"]
        drop_probe = g.loc["plain", "probe"] - g.loc["adversarial", "probe"]
        rows[run] = {
            "kind": g["kind"].iloc[0],
            "drop_model": drop_model,
            "drop_probe": drop_probe,
            "ratio": drop_probe / drop_model,
            "probe_drops_less": bool(drop_probe < drop_model),
        }
    return pd.DataFrame(rows).T


def tier_summary(results) -> pd.DataFrame:
    t = tier_table(results)
    s = shown(t.groupby(["kind", "tier"])[["model", "probe"]].mean())
    s["margin"] = s["probe"] - s["model"]
    s["probe_ahead"] = t.groupby(["kind", "tier"])["margin"].apply(
        lambda m: int((m > 0).sum())
    )
    s["n_runs"] = t.groupby(["kind", "tier"]).size()
    drop = tier_drop(results)
    for kind, g in drop.groupby("kind"):
        s.loc[(kind, "drop"), ["model", "probe"]] = (
            s.loc[(kind, "plain"), ["model", "probe"]].to_numpy()
            - s.loc[(kind, "adversarial"), ["model", "probe"]].to_numpy()
        )
        s.loc[(kind, "drop"), "probe_ahead"] = int(g["probe_drops_less"].sum())
        s.loc[(kind, "drop"), "n_runs"] = len(g)
    order = ["ALL", *CONSTRUCTIONS, "drop"]
    return s.reindex([(k, o) for k in KINDS for o in order if (k, o) in s.index])


def peak_layer(z) -> int:
    """The optimal layer of a run"""
    correct = z["oof"] == z["y"]
    mask = np.ones(len(z["y"]), bool)
    folds = z["folds"]
    per_fold = np.stack(
        [correct[:, mask & (folds == f)].mean(1) for f in range(N_FOLDS)], axis=1
    )
    return int(per_fold.mean(1).argmax())


# relative depth

DEPTH_GRID = np.linspace(0.0, 1.0, 51)
PLATEAU = 0.95


def depths(d) -> np.ndarray:
    """Layer index rescaled to [0, 1], so runs with different layer counts share an axis."""
    return np.arange(d["n_layers"]) / (d["n_layers"] - 1)


def depth_scale(d, col: str = "acc") -> np.ndarray:
    """The layer curve as a share of that run's own best, above chance."""
    acc = d["by_layer"][col].to_numpy()
    top = acc.max()
    return (acc - CHANCE) / (top - CHANCE) if top > CHANCE else np.zeros_like(acc)


def depth_curve(results, col: str = "acc", grid=DEPTH_GRID) -> pd.DataFrame:
    """One rescaled curve per run, interpolated onto a common depth grid."""
    return pd.DataFrame(
        {
            label: np.interp(grid, depths(d), depth_scale(d, col))
            for label, d in results.items()
        },
        index=pd.Index(grid, name="depth"),
    )


def peak_depths(results, col: str = "acc") -> pd.DataFrame:
    """Where each run peaks, in relative depth."""
    rows = {}
    for label, d in results.items():
        x, z = depths(d), depth_scale(d, col)
        above = x[z >= PLATEAU]
        fold_peaks = layer_folds(d).argmax(0) / (d["n_layers"] - 1)
        rows[label] = {
            "n_layers": d["n_layers"],
            "peak_layer": int(z.argmax()),
            "peak_depth": float(x[z.argmax()]),
            "plateau_lo": float(above.min()),
            "plateau_hi": float(above.max()),
            "fold_lo": float(fold_peaks.min()),
            "fold_hi": float(fold_peaks.max()),
            "acc": float(d["by_layer"][col].max()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


DEPTH_BAND = (0.5, 0.8)


def depth_summary(results, col: str = "acc", band=DEPTH_BAND) -> pd.Series:
    """Where the peaks sit, across the runs given: the spread and how many fall in band"""
    peaks = peak_depths(results, col)["peak_depth"]
    inside = (peaks >= band[0]) & (peaks <= band[1])
    return pd.Series(
        {
            "n_runs": len(peaks),
            "min": peaks.min(),
            "q1": peaks.quantile(0.25),
            "median": peaks.median(),
            "q3": peaks.quantile(0.75),
            "max": peaks.max(),
            "band_lo": band[0],
            "band_hi": band[1],
            "in_band": int(inside.sum()),
            "share_in_band": inside.mean(),
        }
    )


def slices(d):
    return [("ALL", np.ones(len(d["y"]), bool))] + [
        (c, d["construction"] == c) for c in d["constructions"]
    ]


def probe_fold_accuracy(d, mask):
    return layer_folds(d, mask)[peak_layer(d)]


def layer_folds(d, mask=None):
    mask = np.ones(len(d["y"]), bool) if mask is None else mask
    return np.stack(
        [d["correct"][:, mask & (d["folds"] == f)].mean(1) for f in range(N_FOLDS)],
        axis=1,
    )


def compare(d) -> pd.DataFrame:
    """Accuracy per router per construction, all on the same folds."""
    per_fold = compare_folds(d)
    rows = {}
    for name, mask in slices(d):
        row = {r: float(np.mean(per_fold.loc[name, r])) for r in ROUTERS}
        row["peak"] = peak_layer(d)
        row["n"] = int(mask.sum())
        rows[name] = row
    return pd.DataFrame(rows).T


def compare_folds(d) -> pd.DataFrame:
    correct = d["predicted"] == d["y"]
    per_fold = d["baselines"].get("per_fold", {})
    rows = {}
    for name, mask in slices(d):
        rows[name] = {
            "model": fold_accuracy(correct, d["folds"], mask),
            "lexical": np.asarray(
                per_fold.get("lexical", {}).get(name, [np.nan] * N_FOLDS), float
            ),
            "tfidf": np.asarray(
                per_fold.get("tfidf", {}).get(name, [np.nan] * N_FOLDS), float
            ),
            "embedding": np.asarray(
                per_fold.get("embedding", {}).get(name, [np.nan] * N_FOLDS), float
            ),
            "probe": probe_fold_accuracy(d, mask),
        }
    return pd.DataFrame(rows).T


def significance(d) -> pd.DataFrame:
    from statsmodels.stats.contingency_tables import mcnemar

    model_right = d["predicted"] == d["y"]
    rows = {}
    peak = peak_layer(d)
    for name, mask in slices(d):
        probe_right = d["correct"][peak]
        both = int((model_right & probe_right & mask).sum())
        model_only = int((model_right & ~probe_right & mask).sum())
        probe_only = int((~model_right & probe_right & mask).sum())
        neither = int((~model_right & ~probe_right & mask).sum())
        table = [[both, model_only], [probe_only, neither]]
        rows[name] = {
            "n": int(mask.sum()),
            "peak": peak,
            "model_only": model_only,
            "probe_only": probe_only,
            "p_mcnemar": float(mcnemar(table, exact=True).pvalue),  # type: ignore
        }
    return pd.DataFrame(rows).T


def holm(p: pd.Series) -> pd.Series:
    from statsmodels.stats.multitest import multipletests

    return pd.Series(multipletests(p.astype(float), method="holm")[1], index=p.index)


def significance_family(site: str = "prompt") -> pd.DataFrame:
    probed = [
        (m, k, site) for m, k in (runs() if site == "prompt" else reasoning_runs())
    ]
    rows = {}
    for model, kind, site in probed:
        s = significance(result(model, kind, site))
        rows[(model, kind, site)] = {
            "model_only_all": s.loc["ALL", "model_only"],
            "probe_only_all": s.loc["ALL", "probe_only"],
            "p_all": s.loc["ALL", "p_mcnemar"],
            "model_only_adv": s.loc["adversarial", "model_only"],
            "probe_only_adv": s.loc["adversarial", "probe_only"],
            "p_adv": s.loc["adversarial", "p_mcnemar"],
        }
    t = pd.DataFrame(rows).T
    t.index.names = ["model", "kind", "site"]
    adj = holm(pd.concat([t["p_all"], t["p_adv"]], keys=["all", "adv"]))
    t["p_all_holm"], t["p_adv_holm"] = adj["all"].to_numpy(), adj["adv"].to_numpy()
    return t


def run_tables(
    site: str = "prompt", family: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The chapter-7 per-model tables, one row per run probed at last prompt/reasoning token."""
    family = significance_family(site) if family is None else family
    rows = {}
    for model, kind, _ in family.index:
        d = result(model, kind, site)
        proba, layer = probabilities(model, kind, site=site)
        c, f = shown(compare(d).astype(float)), family.loc[(model, kind, site)]
        row = {"layer": f"{layer}/{d['n_layers']}", "auc_macro": auc_macro(d, proba)}
        for sl, key in [("all", "ALL"), ("adv", "adversarial")]:
            row |= {
                f"model_{sl}": c.loc[key, "model"],
                f"probe_{sl}": c.loc[key, "probe"],
                f"delta_{sl}": c.loc[key, "probe"] - c.loc[key, "model"],
                f"b_{sl}": f[f"model_only_{sl}"],
                f"c_{sl}": f[f"probe_only_{sl}"],
                f"p_{sl}": f[f"p_{sl}"],
                f"p_{sl}_holm": f[f"p_{sl}_holm"],
            }
        rows[(model, kind, site)] = row
    t = pd.DataFrame(rows).T
    t.index.names = ["model", "kind", "site"]
    return t


def site_table(model: str, kind: str) -> pd.DataFrame:
    """Probe at the last prompt token against probe at the last reasoning token"""
    return pd.concat([run_tables(site).loc[[(model, kind, site)]] for site in SITES])


def decision_readout(model: str, kind: str) -> pd.DataFrame:
    """Does the hidden state hold the request or the model's own choice? One row per site."""
    rows = {}
    for site in SITES:
        d, z = load(model, kind, site), result(model, kind, site)
        layer = peak_layer(z)
        probe, predicted, y = z["oof"][layer], z["predicted"], z["y"]
        model_ok, probe_ok = predicted == y, probe == y
        routed = ~pd.isna(predicted)
        dr = {
            **d,
            "y": predicted[routed].astype(object),
            "H": d["H"][routed],
            "folds": d["folds"][routed],
        }
        decision = probe_predictions(dr, layer)
        rows[site] = {
            "layer": layer,
            "agreement": float((probe == predicted).mean()),
            "both": int((model_ok & probe_ok).sum()),
            "model_only": int((model_ok & ~probe_ok).sum()),
            "probe_only": int((~model_ok & probe_ok).sum()),
            "neither": int((~model_ok & ~probe_ok).sum()),
            "decision_acc": float((decision == dr["y"]).mean()),
            "decision_vs_gold": float((decision == y[routed]).mean()),
        }
    return pd.DataFrame(rows).T


def flags(d, layer: int | None = None) -> pd.DataFrame:
    """The probe as a misroute detector rather than as a router."""
    layer = peak_layer(d) if layer is None else layer
    probe = d["oof"][layer]
    right = probe == d["y"]
    wrong = d["predicted"] != d["y"]
    rows = {}
    for name, mask in [("ALL", np.ones(len(d["y"]), bool))] + [
        (c, d["construction"] == c) for c in d["constructions"]
    ]:
        flagged = (probe != d["predicted"]) & mask
        precision = wrong[flagged].mean() if flagged.any() else np.nan
        recall = (flagged & wrong).sum() / max(wrong[mask].sum(), 1)
        rows[name] = {
            "flagged": int(flagged.sum()),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else np.nan,
            "base_rate": wrong[mask].mean(),
            "false_alarm": (flagged & ~wrong).sum() / max((~wrong & mask).sum(), 1),
            # a deployment that sends every flagged sample to a fallback:
            # the share of traffic it reviews, and the error left unreviewed
            "flagged_share": flagged.sum() / mask.sum(),
            "residual_error": (wrong & ~flagged & mask).sum() / mask.sum(),
            "margin": shown(right[mask].mean()) - shown((~wrong[mask]).mean()),
            "layer": layer,
        }
    return pd.DataFrame(rows).T


FLAG_METRICS = ("precision", "recall", "f1", "false_alarm")


def flag_correlations(results, slice_: str = "ALL") -> pd.DataFrame:
    """How each flag metric moves with the probe's margin over the model, across runs."""
    table = across_runs(flags, slice_=slice_, results=results).astype(float)
    margin = table["margin"]
    return pd.DataFrame(
        {
            m: {
                "r_with_margin": float(np.corrcoef(margin, table[m])[0, 1]),
                "min": float(table[m].min()),
                "max": float(table[m].max()),
            }
            for m in FLAG_METRICS
        }
    ).T


# failure analysis

CELLS = ("both", "probe_only", "baseline_only", "neither")
FACTORS = (
    "construction",
    "gold",
    "collision",
    "decoy",
    "persona",
    "pattern",
    "request_id",
)
DEPENDENCE_TIERS = ("natural", "adversarial")


def cells(d, layer: int | None = None, baseline: str = "lexical") -> pd.DataFrame:
    """One row per sample: who got it right, the probe or the text baseline, and on what."""
    layer = peak_layer(d) if layer is None else layer
    b = d["baselines"]
    ids = list(d["ids"])
    base = pd.Series(b["pred"][baseline], index=b["ids"]).reindex(ids).to_numpy()
    out = pd.DataFrame(
        {
            "id": ids,
            "gold": d["y"],
            "probe": d["oof"][layer],
            "baseline": base,
            "model": d["predicted"],
            "fold": d["folds"],
        }
    )
    factors = dataset(d["kind"]).set_index("id").reindex(ids)
    for f in FACTORS:
        if f in factors:
            out[f] = factors[f].to_numpy()
    probe_ok = out["probe"].to_numpy() == out["gold"].to_numpy()
    base_ok = out["baseline"].to_numpy() == out["gold"].to_numpy()
    out["probe_ok"] = probe_ok
    out["baseline_ok"] = base_ok
    out["model_ok"] = out["model"].to_numpy() == out["gold"].to_numpy()
    out["cell"] = np.select(
        [probe_ok & base_ok, probe_ok & ~base_ok, ~probe_ok & base_ok],
        ["both", "probe_only", "baseline_only"],
        default="neither",
    )
    return out


def dependence(d, baseline: str = "lexical") -> pd.DataFrame:
    """Does the probe's correctness depend on the baseline's, tier by tier? One run."""
    c = cells(d, baseline=baseline)
    rows = {}
    for tier in DEPENDENCE_TIERS:
        t = c[c["construction"] == tier]
        base_ok = t["baseline_ok"].to_numpy()
        ok = t["probe_ok"].to_numpy()
        right, wrong = ok[base_ok].mean(), ok[~base_ok].mean()
        kappa = cohen_kappa_score(ok, base_ok)
        p1, p2 = ok.mean(), base_ok.mean()
        p_e = p1 * p2 + (1 - p1) * (1 - p2)
        kappa_ceiling = (1 - abs(p1 - p2) - p_e) / (1 - p_e)
        rows[tier] = {
            "n": len(t),
            "baseline_wrong": int((~base_ok).sum()),
            "baseline_acc": base_ok.mean(),
            "acc": ok.mean(),
            "acc_baseline_right": right,
            "acc_baseline_wrong": wrong,
            "gap": wrong - right,
            "kappa": kappa,
            "kappa_ceiling": kappa_ceiling,
            "kappa_ratio": kappa / kappa_ceiling,
        }
    return pd.DataFrame(rows).T.rename_axis("tier")


def dependence_table(results, names=BASELINE_NAMES) -> pd.DataFrame:
    t = pd.concat(
        {(label, b): dependence(d, b) for label, d in results.items() for b in names},
        names=["run", "baseline"],
    )
    t["kind"] = [split_label(r)[1] for r in t.index.get_level_values("run")]
    return t


def dependence_summary(results, names=BASELINE_NAMES) -> pd.DataFrame:
    t = dependence_table(results, names)
    keys = ["kind", "baseline", "tier"]
    t = t.reset_index().astype(
        {
            c: float
            for c in [
                "n",
                "baseline_wrong",
                "baseline_acc",
                "acc",
                "acc_baseline_right",
                "acc_baseline_wrong",
                "gap",
                "kappa",
                "kappa_ceiling",
                "kappa_ratio",
            ]
        }
    )
    g = t.groupby(keys, sort=False)
    s = g[
        [
            "baseline_acc",
            "acc",
            "acc_baseline_right",
            "acc_baseline_wrong",
            "gap",
            "kappa",
            "kappa_ceiling",
            "kappa_ratio",
        ]
    ].mean()
    cols = ["acc_baseline_right", "acc_baseline_wrong"]
    s[cols] = shown(s[cols])
    s["gap"] = s["acc_baseline_wrong"] - s["acc_baseline_right"]
    s["gap_min"], s["gap_max"] = g["gap"].min(), g["gap"].max()
    s["kappa_min"], s["kappa_max"] = g["kappa"].min(), g["kappa"].max()
    s["n_runs"] = g.size()
    return s


def pair_outcomes(d, names=BASELINE_NAMES) -> pd.DataFrame:
    """How every router handles the adversarial pairs of one run, one row per router."""
    frames = {"probe": None, "model": None, **{b: None for b in names}}
    base = cells(d, baseline=names[0])
    pair_sizes = (
        base[base["construction"] == "adversarial"].groupby("request_id").size()
    )
    complete = pair_sizes[pair_sizes == 2].index
    rows = {}
    for router in frames:
        if router in names:
            c = cells(d, baseline=router)
            pred = c["baseline"]
        else:
            c = base
            pred = c[router]
        t = c.assign(pred=pred, ok=pred == c["gold"])
        t = t[t["construction"].eq("adversarial") & t["request_id"].isin(complete)]
        g = t.groupby("request_id")
        right = g["ok"].sum()
        rows[router] = {
            "pairs": len(right),
            "both": (right == 2).mean(),
            "one": (right == 1).mean(),
            "neither": (right == 0).mean(),
            "collapsed": (g["pred"].nunique() == 1).mean(),
        }
    return pd.DataFrame(rows).T


def pair_table(results, names=BASELINE_NAMES) -> pd.DataFrame:
    t = pd.concat(
        {label: pair_outcomes(d, names) for label, d in results.items()},
        names=["run", "router"],
    ).astype(float)
    t["kind"] = [split_label(r)[1] for r in t.index.get_level_values("run")]
    return t


def pair_summary(results, names=BASELINE_NAMES) -> pd.DataFrame:
    t = pair_table(results, names).reset_index()
    g = t.groupby(["kind", "router"], sort=False)
    s = g[["pairs", "both", "one", "neither", "collapsed"]].mean()
    s["both_min"], s["both_max"] = g["both"].min(), g["both"].max()
    s["collapsed_min"], s["collapsed_max"] = g["collapsed"].min(), g["collapsed"].max()
    s["n_runs"] = g.size()
    return s


def error_direction(d, names=BASELINE_NAMES) -> pd.DataFrame:
    """How often a router predicts the decoy, one row per (router, tier) with a decoy."""
    base = cells(d, baseline=names[0])
    rows = {}
    for router in ("probe", "model", *names):
        pred = (
            cells(d, baseline=router)["baseline"] if router in names else base[router]
        )
        wrong = pred != base["gold"]
        on_decoy = pred == base["decoy"]
        for tier in DEPENDENCE_TIERS:
            m = base["construction"] == tier
            rows[(router, tier)] = {
                "n": int(m.sum()),
                "error": float(wrong[m].mean()),
                "decoy": float(on_decoy[m].mean()),
                "other": float((wrong & ~on_decoy)[m].mean()),
                "decoy_share": float(on_decoy[m & wrong].mean()),
            }
    return pd.DataFrame(rows).T.rename_axis(["router", "tier"])


def error_direction_table(results, names=BASELINE_NAMES) -> pd.DataFrame:
    t = pd.concat(
        {label: error_direction(d, names) for label, d in results.items()},
        names=["run"],
    ).astype(float)
    t["kind"] = [split_label(r)[1] for r in t.index.get_level_values("run")]
    return t


def error_direction_summary(results, names=BASELINE_NAMES) -> pd.DataFrame:
    t = error_direction_table(results, names).reset_index()
    g = t.groupby(["kind", "router", "tier"], sort=False)
    s = g[["error", "decoy", "other", "decoy_share"]].mean()
    s["decoy_min"], s["decoy_max"] = g["decoy"].min(), g["decoy"].max()
    probe = t[t["router"] == "probe"].set_index(["run", "tier"])["decoy"]
    below = t.assign(
        probe=probe.reindex(pd.MultiIndex.from_frame(t[["run", "tier"]])).to_numpy()
    )
    s["probe_below"] = (
        (below["probe"] < below["decoy"])
        .groupby([below["kind"], below["router"], below["tier"]], sort=False)
        .sum()
    )
    s["n_runs"] = g.size()
    return s


def pooled_runs(results) -> dict:
    """The runs every cross-run view of chapter 8 is computed on (gpt-oss-20b left out)"""
    return {k: v for k, v in results.items() if "gpt-oss" not in k}


def across_runs(fn, slice_="ALL", results=None) -> pd.DataFrame:
    results = results or {f"{m} {k}": result(m, k) for m, k in runs()}
    return pd.DataFrame({label: fn(d).loc[slice_] for label, d in results.items()}).T


def multi_turn_cost(results=None) -> pd.DataFrame:
    """Per model, how much accuracy the multi-turn setting costs the model and the probe."""
    results = results or {f"{m} {k}": result(m, k) for m, k in runs()}
    acc = shown(across_runs(compare, results=results)[["model", "probe"]].astype(float))
    models = sorted({m for m, _ in runs()})
    rows = {}
    for m in models:
        single, multi = f"{m} single_turn", f"{m} multi_turn"
        if single not in acc.index or multi not in acc.index:
            continue
        delta_model = acc.loc[multi, "model"] - acc.loc[single, "model"]
        delta_probe = acc.loc[multi, "probe"] - acc.loc[single, "probe"]
        rows[m] = {
            "model_single": acc.loc[single, "model"],
            "model_multi": acc.loc[multi, "model"],
            "delta_model": delta_model,
            "probe_single": acc.loc[single, "probe"],
            "probe_multi": acc.loc[multi, "probe"],
            "delta_probe": delta_probe,
            "ratio": abs(delta_probe) / abs(delta_model) if delta_model else np.nan,
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    import sys

    wanted = sys.argv[1:]
    for model, kind in runs():
        if wanted and model not in wanted:
            continue
        sweep(model, kind)
        probabilities(model, kind)
        print(f"done {model} {kind}", flush=True)
