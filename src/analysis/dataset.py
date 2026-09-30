# https://www.statskingdom.com/doc_anova.html

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from analysis import probe as P
from analysis.paths import DATA, MODERNBERT_FROZEN

FACTORS = [
    "gold",
    "decoy",
    "construction",
    "pattern",
    "persona",
    "collision",
    "variant",
]
# collision is dropped from the one-way family: it is defined as the pair (gold, decoy),
# so its effect is not a new fact next to those two factors' own effects
ONEWAY_FACTORS = [f for f in FACTORS if f != "collision"]
WORD_RE = re.compile(r"[a-zA-Z']+")

# a magnitude can be written without a digit ("eighty percent", "two weeks"), so pct_digit
# alone does not measure how numeric a query is. these two do the rest of the job.
UNIT_RE = re.compile(
    r"%|\$|€|(?<![a-z])(?:ms|sec|secs|min|mins|hr|hrs|[kmgt]b|[kmg]ib|rps|qps|req/s|ops/s)\b",
    re.I,
)
NUMWORD_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|"
    r"percent|half|quarter|double|triple|dozen)\b",
    re.I,
)
NUMERIC = ["pct_digit", "n_units", "n_number_words"]


def surface(t: str) -> dict:
    words = WORD_RE.findall(t)
    n_words = len(words) or 1
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    return {
        "n_chars": len(t),
        "n_words": len(words),
        "n_sentences": len(sents),
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "mean_sent_len": len(words) / max(len(sents), 1),
        "ttr": len({w.lower() for w in words}) / n_words,  # lexical diversity
        "pct_upper": sum(c.isupper() for c in t) / max(len(t), 1),
        "pct_digit": sum(c.isdigit() for c in t) / max(len(t), 1),
        "n_question": t.count("?"),
        "n_comma": t.count(","),
        "long_word_ratio": sum(len(w) > 7 for w in words) / n_words,
        "n_units": len(UNIT_RE.findall(t)),
        "n_number_words": len(NUMWORD_RE.findall(t)),
    }


def frame(kind: str) -> tuple[pd.DataFrame, list[str]]:
    raw = P.dataset(kind)
    feat = pd.DataFrame([surface(t) for t in raw["user_text"]], index=raw.index)
    d = pd.concat([raw, feat], axis=1)
    d["log_words"] = np.log(d["n_words"])  # length is right-skewed; ANOVA likes the log
    d["has_digit"] = d["pct_digit"] > 0
    return d, list(feat.columns)


def top_terms(kind: str, n: int = 12) -> dict[str, pd.DataFrame]:
    df = P.dataset(kind)
    text, y = df["user_text"].to_numpy(), df["gold"].to_numpy()
    out = {}
    for name, vec in [
        ("lexical", CountVectorizer(ngram_range=(1, 2), min_df=2)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ]:
        X = vec.fit_transform(text)
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        vocab = np.asarray(vec.get_feature_names_out())
        # one-vs-rest coefficients: per expert, the terms pushing hardest towards it
        out[name] = pd.DataFrame(
            {
                e: vocab[np.argsort(row)[::-1][:n]]
                for e, row in zip(clf.classes_, clf.coef_)
            }
        )
    return out


def cramers_v(a, b) -> float:
    tab = pd.crosstab(a, b)
    r, k = tab.shape
    return (
        float(stats.contingency.association(tab, method="cramer"))
        if min(r, k) > 1
        else np.nan
    )


def balance(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [[cramers_v(d[a], d[b]) for b in FACTORS] for a in FACTORS],
        index=FACTORS,
        columns=FACTORS,
    )


def _oneway(frame_, factor, feature):
    groups = [g[feature].values for _, g in frame_.groupby(factor) if len(g) > 1]
    if len(groups) < 2:
        return None

    F, p = stats.f_oneway(*groups)

    grand = frame_[feature].mean()

    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((frame_[feature] - grand) ** 2).sum()

    # eta2 is biased upwards: ss_between also holds the sampling noise of the group means.
    # omega2 subtracts the expected amount of that noise, same scale and same thresholds.
    k, n = len(groups), sum(len(g) for g in groups)
    ms_within = (ss_total - ss_between) / (n - k) if n > k else np.nan
    omega2 = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)

    return {
        "factor": factor,
        "feature": feature,
        "k": len(groups),
        "F": F,
        "p": p,
        "eta2": ss_between / ss_total if ss_total else np.nan,
        "omega2": float(omega2),
        "levene_p": stats.levene(*groups)[1],
        "kruskal_p": stats.kruskal(*groups)[1],
    }


def anova(d: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in ONEWAY_FACTORS:
        # pattern / variant only exist for adversarial rows
        sub = d[d[f] != "none"] if (d[f] == "none").any() else d
        for y in features:
            r = _oneway(sub, f, y)
            if r:
                rows.append(r)
    out = pd.DataFrame(rows)

    # each one-way factor is its own family of thirteen tests, one per feature: gold's
    # p-holm must not depend on decoy's or persona's p-values, since only gold's family
    # is ever reported or interpreted together
    for col, holm_col in [
        ("p", "p_holm"),
        ("kruskal_p", "kruskal_holm"),
        ("levene_p", "levene_holm"),
    ]:
        out[holm_col] = out.groupby("factor")[col].transform(
            lambda s: multipletests(s, method="holm")[1]
        )

    # omega2 is the effect size reported everywhere downstream: eta2 stays in the frame
    # only as the intermediate the correction is computed from
    return out.sort_values("omega2", ascending=False)


def _formulas(factors) -> tuple[str, str]:
    others = [f for f in factors if f != "gold"]
    full = " * ".join(f"C({f})" for f in factors)
    reduced = "C(gold) + " + " * ".join(f"C({f})" for f in others)
    return full, reduced


def _nested_test(d, y, full, reduced) -> dict:
    fit_reduced = smf.ols(f"{y} ~ {reduced}", data=d).fit()
    fit_full = smf.ols(f"{y} ~ {full}", data=d).fit()
    tab = sm.stats.anova_lm(fit_reduced, fit_full)
    df_diff = int(tab["df_diff"].iloc[1])
    ss_diff = fit_reduced.ssr - fit_full.ssr
    # same correction as omega2 one-way, on the block of interaction terms instead of a
    # single term: ss_diff / df_diff replace ss_A / df_A, the error comes from the full model.
    # it can go below zero, which means "not distinguishable from noise"
    ms_error = fit_full.ssr / fit_full.df_resid
    n = len(d)
    return {
        "feature": y,
        # ssr -> sum of squared residuals, aka sum of squared errors
        # dividing by the error the reduced model left (ss_diff + ssr_full == ssr_reduced)
        # makes every feature top out at 1, so the column can be ranked across features
        # https://resources.nu.edu/statsresources/eta
        "partial_eta2": ss_diff / (ss_diff + fit_full.ssr),
        # https://www.aggieerin.com/shiny-server/tests/omegapbnss.html
        "partial_omega2": float(
            (ss_diff - df_diff * ms_error) / (ss_diff + (n - df_diff) * ms_error)
        ),
        "df": df_diff,
        "F": float(tab["F"].iloc[1]),
        "p": float(tab["Pr(>F)"].iloc[1]),
    }


def interactions(
    d: pd.DataFrame,
    features: list[str],
    factors=("gold", "construction", "persona"),
) -> pd.DataFrame:
    d = d[~(d[list(factors)] == "none").any(axis=1)]

    full, reduced = _formulas(factors)
    out = pd.DataFrame([_nested_test(d, y, full, reduced) for y in features])

    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    return out.sort_values("partial_omega2", ascending=False)


def posthoc(
    d: pd.DataFrame, anova_: pd.DataFrame, features=None, alpha=0.05, min_omega2=0.01
):
    """
    Games-Howell on every feature of the family, including the ones where Levene fails:
    """
    gold = anova_[anova_.factor == "gold"]
    if features is None:
        keep = gold[(gold["p_holm"] < alpha) & (gold["omega2"] >= min_omega2)]
        features = list(keep.sort_values("omega2", ascending=False)["feature"])
    experts = sorted(d["gold"].unique())

    rows = []
    for f in features:
        groups = [d.loc[d["gold"] == e, f].to_numpy() for e in experts]
        res = stats.tukey_hsd(*groups, equal_var=False)
        ci = res.confidence_interval()
        rows += [
            {
                "feature": f,
                "a": a,
                "b": b,
                "diff": float(res.statistic[i, j]),  # mean(a) - mean(b)
                "ci_low": float(ci.low[i, j]),
                "ci_high": float(ci.high[i, j]),
                "p": float(res.pvalue[i, j]),
            }
            for i, a in enumerate(experts)
            for j, b in enumerate(experts)
            if i < j
        ]

    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    out["feature"] = pd.Categorical(out["feature"], categories=features, ordered=True)
    return out.sort_values(["feature", "p"], ignore_index=True)


def ceiling(d: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Accuracy from the surface features alone, out of fold, per construction."""
    y = d["gold"].to_numpy()
    construction = d["construction"].to_numpy()
    folds = P.folds_for(d["scenario_id"].to_numpy(), y)

    def row(clf, cols):
        X = d[cols].to_numpy()
        pred = P.out_of_fold(
            folds, y, lambda tr, te: clone(clf).fit(X[tr], y[tr]).predict(X[te])
        )
        correct = pred == y
        return {"ALL": float(correct.mean())} | {
            c: float(correct[construction == c].mean())
            for c in P.CONSTRUCTIONS
            if (construction == c).any()
        }

    logreg = make_pipeline(StandardScaler(), P.logreg())
    no_digit = [f for f in features if f != "pct_digit"]
    numeric = [f for f in NUMERIC if f in features]
    no_numeric = [f for f in features if f not in numeric]
    rf = RandomForestClassifier(n_estimators=400, random_state=P.SEED, n_jobs=-1)

    return pd.DataFrame(
        {
            "logreg, all features": row(logreg, features),
            "logreg, without pct_digit": row(logreg, no_digit),
            "logreg, pct_digit only": row(logreg, ["pct_digit"]),
            "logreg, without numeric": row(logreg, no_numeric),
            "logreg, numeric only": row(logreg, numeric),
            "random forest, all features": row(rf, features),
        }
    ).T


def importances(d: pd.DataFrame, features: list[str]) -> pd.Series:
    rf = RandomForestClassifier(n_estimators=400, random_state=P.SEED, n_jobs=-1)
    rf.fit(d[features].values, d["gold"].values)  # type: ignore
    return pd.Series(rf.feature_importances_, index=features).sort_values()


def shap_importance(d: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature and per expert, out of fold."""
    import shap

    X = d[features].to_numpy(dtype=float)
    y = d["gold"].to_numpy()
    folds = P.folds_for(d["scenario_id"].to_numpy(), y)
    experts = [e for e in P.EXPERTS if e in set(y)]

    total = np.zeros((len(features), len(experts)))
    for f in range(P.N_FOLDS):
        held_out = folds == f
        rf = RandomForestClassifier(n_estimators=400, random_state=P.SEED, n_jobs=-1)
        rf.fit(X[~held_out], y[~held_out])
        values = shap.TreeExplainer(rf).shap_values(X[held_out])
        # shap >= 0.45 returns one array (rows, features, classes); older versions
        # return a list of (rows, features), one per class
        if isinstance(values, list):
            values = np.stack(values, axis=-1)
        # rf.classes_ is alphabetical, `experts` is taxonomy order: reorder by name
        order = [list(rf.classes_).index(e) for e in experts]
        total += np.abs(values[:, :, order]).sum(axis=0)

    out = pd.DataFrame(total / len(y), index=features, columns=experts)
    return out.loc[out.sum(axis=1).sort_values().index]


def pair_overlaps() -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "kind": k,
                    "pair_overlap": P.dataset(k)
                    .query("construction == 'adversarial'")["pair_overlap"]
                    .astype(float),
                }
            )
            for k in P.KINDS
        ]
    )


def _by_construction(correct, construction, name) -> pd.Series:
    """Pooled accuracy on the whole kind and inside each construction."""
    return pd.Series(
        {"ALL": float(correct.mean())}
        | {c: float(correct[construction == c].mean()) for c in P.CONSTRUCTIONS},
        name=name,
    )


# the two encoders below are hardness checks on the dataset, next to the embedding
# baseline: both are kept out of `P.BASELINE_NAMES`, so the probe comparisons never see
# them, and both are pooled over the out-of-fold predictions, as `P.baselines`


def frozen_encoder(kind: str) -> pd.Series:
    """Accuracy of frozen ModernBERT-large + logistic regression, per construction,
    read from simulate/modernbert_frozen_app.py output."""
    rows = pd.read_csv(MODERNBERT_FROZEN / f"{kind}.csv")
    d = P.dataset(kind)
    rows = rows.set_index("id").loc[d["id"]]
    folds = P.folds_for(d["scenario_id"].to_numpy(), d["gold"].to_numpy())
    assert (rows["fold"].to_numpy() == folds).all(), "not the probes' folds"
    assert (rows["gold"].to_numpy() == d["gold"].to_numpy()).all()
    correct = rows["predicted"].to_numpy() == d["gold"].to_numpy()
    return _by_construction(correct, d["construction"].to_numpy(), "modernbert")


M3_MODEL = "BAAI/bge-m3"


def m3_embeddings(kind: str) -> np.ndarray:
    """bge-m3 dense embeddings of `user_text`, cached on disk per kind. Its 8192-token
    window holds every transcript, so nothing is truncated."""
    d = P.dataset(kind)
    path = DATA / f".bge-m3_{kind}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if z["ids"].tolist() == d["id"].tolist():
            return z["X"]
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(M3_MODEL)
    X = np.asarray(
        encoder.encode(
            d["user_text"].tolist(),
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )
    np.savez(path, X=X, ids=np.array(d["id"], dtype=object))
    return X


def m3_encoder(kind: str) -> pd.Series:
    """Accuracy of bge-m3 embeddings + logistic regression, per construction: the
    embedding baseline with a larger encoder, on the same folds."""
    d = P.dataset(kind)
    y = d["gold"].to_numpy()
    folds = P.folds_for(d["scenario_id"].to_numpy(), y)
    X = m3_embeddings(kind)
    pred = P.out_of_fold(
        folds, y, lambda tr, te: P.logreg().fit(X[tr], y[tr]).predict(X[te])
    )
    return _by_construction(pred == y, d["construction"].to_numpy(), "bge-m3")
