import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # The dataset

    Five experts. In each sample one constraint is binding, and the label is the expert that owns it.

    This notebook checks if surface features (length, punctuation, digits) predict the label.
    If they do, the text already contains the answer and probing adds nothing.
    Data: `dataset/<kind>.csv`. Pick the kind below.
    """)
    return (mo,)


@app.cell
def _():
    import warnings

    import pandas as pd

    from analysis import dataset as dslib
    from analysis import plots
    from analysis import probe as P

    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 160, "display.max_columns", 50)
    return P, dslib, pd, plots


@app.cell
def _(mo):
    kind = mo.ui.dropdown(["single_turn", "multi_turn"], value="single_turn", label="kind")
    kind # type: ignore
    return (kind,)


@app.cell
def _(dslib, kind):
    d, FEATURES = dslib.frame(kind.value)
    print(f"{kind.value}: {len(d)} samples, {d['scenario_id'].nunique()} scenarios")
    d[FEATURES].describe().T.round(3)
    return FEATURES, d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Composition

    Each expert has the same number of samples in each stratum.
    """)
    return


@app.cell
def _(kind, plots):
    plots.plot_composition(kind.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Design balance

    Cramér's V between the design factors. Read the `gold` row: a value above 0 against
    `construction`, `persona` or `pattern` means the label is tied to that factor.
    `collision`, `decoy` and `variant` depend on `gold` by definition.
    """)
    return


@app.cell
def _(d, dslib, plots):
    V = dslib.balance(d)
    plots.plot_balance(V)
    return (V,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. One-way ANOVA

    Each factor against each surface feature. n is large, so read `omega2` (effect size), not p.
    Guide: 0.01 small, 0.06 medium, 0.14 large.

    `collision` is left out: it is the pair (`gold`, `decoy`).
    """)
    return


@app.cell
def _(FEATURES, d, dslib):
    anova = dslib.anova(d, FEATURES)
    anova.head(15).round(4)
    return (anova,)


@app.cell
def _(anova, plots):
    plots.plot_omega2(anova)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `gold` rows, largest effect first.
    """)
    return


@app.cell
def _(anova):
    anova[anova.factor == "gold"].sort_values("omega2", ascending=False)[
        ["feature", "omega2", "p_holm"]
    ].round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `gold` against the style factors

    Style factors (`persona`, `construction`, `pattern`) can change the surface features.
    A shortcut exists only if `gold` changes them.
    """)
    return


@app.cell
def _(P, dslib, plots):
    _frames = {k: dslib.frame(k) for k in P.KINDS}
    _data = {k: d for k, (d, _) in _frames.items()}
    _anovas = {k: dslib.anova(d, features) for k, (d, features) in _frames.items()}
    plots.plot_gold_effect(_data, _anovas)
    return


@app.cell
def _(anova):
    # collision is not in the one-way family (dslib.ONEWAY_FACTORS): it is defined as the
    # pair (gold, decoy), so its effect is not a new fact next to theirs
    anova[anova.factor.isin(["gold", "decoy"])][
        # kruskal_holm and levene_holm, not their raw p-values: comparing a corrected
        # verdict against a raw one invents disagreements that are only different
        # thresholds
        ["factor", "feature", "k", "omega2", "p_holm", "kruskal_holm", "levene_holm"]
    ].round(4)
    return


@app.cell
def _(d):
    # digits are not a routing constraint: if one expert's queries simply contain more
    # numbers, that is a shortcut. Checked as the mandate and as the mere presence of the
    # constraint in the text. n_units is the same shortcut written without digits.
    print(d.groupby("gold")[
        ["pct_digit", "has_digit", "n_units", "n_number_words", "mean_word_len", "long_word_ratio"]
    ].mean().round(3))
    print("\nP(digit) by whether resource_opt is involved at all (gold or decoy):")
    print(d.groupby((d.gold == "resource_opt") | (d.decoy == "resource_opt"))["has_digit"].mean().round(3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Post-hoc: which expert pairs differ

    Games-Howell (no equal-variance assumption) on the raw values, so `diff` is in the
    feature's units. Only features whose `gold` effect passes Holm and has omega2 of 0.01 or more.
    Holm across features.
    """)
    return


@app.cell
def _(anova, d, dslib):
    post = dslib.posthoc(d, anova)
    print(f"{post['feature'].nunique()} features x 10 pairs = {len(post)} tests, "
          f"{(post['p_holm'] < 0.05).sum()} significant after Holm")
    post[post["p_holm"] < 0.05].round(4)
    return (post,)


@app.cell
def _(kind, plots, post):
    plots.plot_posthoc(post, kind=kind.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Interactions

    One-way tests miss an effect that changes sign between strata. Each fit compares `reduced`
    (the `gold` main effect and every term without `gold`) with `full` (adds the `gold` interactions).

    - `partial_omega2`: gain of `full` over `reduced`, corrected for the extra terms. Rank on this. Below 0 means noise.
    - `partial_eta2`: the same gain, not corrected.
    - `p` / `p_holm`: partial F-test, Holm across features.
    """)
    return


@app.cell
def _(FEATURES, d, dslib):
    # the style knobs
    dslib.interactions(d, FEATURES).round(4)
    return


@app.cell
def _(FEATURES, d, dslib):
    # what makes a sample hard: `collision` is the (gold, decoy) pair, so `gold * decoy`
    # already spans it, so adding it too would only make the model rank-deficient
    dslib.interactions(d, FEATURES, ("gold", "decoy", "construction")).round(4)
    return


@app.cell
def _(FEATURES, d, dslib):
    # the adversarial template. `pattern` is `none` outside that stratum, so the fit
    # restricts itself to the 900 adversarial rows
    dslib.interactions(d, FEATURES, ("gold", "pattern")).round(4)
    return


@app.cell
def _(FEATURES, d, dslib, plots):
    interaction_models = {
        "style": dslib.interactions(d, FEATURES),
        "difficulty": dslib.interactions(d, FEATURES, ("gold", "decoy", "construction")),
        "pattern": dslib.interactions(d, FEATURES, ("gold", "pattern")),
    }
    plots.plot_interactions(interaction_models)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Ceiling: surface features only

    A classifier trained on the surface features, without the text. Same scenario-grouped folds
    as the text baselines. Chance is 0.20. Read the `adversarial` row.
    """)
    return


@app.cell
def _(FEATURES, d, dslib, kind, plots):
    ceiling = dslib.ceiling(d, FEATURES)
    print(ceiling.round(3))
    plots.plot_ceiling(ceiling, kind.value)
    return (ceiling,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SHAP importance per expert, out of fold, on the same folds.
    """)
    return


@app.cell
def _(FEATURES, d, dslib, kind, plots):
    shap_imp = dslib.shap_importance(d, FEATURES)
    print(shap_imp.round(4))
    plots.plot_shap(shap_imp, kind.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Text baselines

    Classifiers that read only the text, on the same folds. The probe is compared with these.
    """)
    return


@app.cell
def _(P, pd, plots):
    text_baselines = {k: pd.DataFrame(P.baselines(k)["accuracy"]) for k in P.KINDS}
    print(pd.concat(text_baselines, axis=1).round(3))
    plots.plot_baselines()
    return (text_baselines,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two more frozen encoders next to the embedding baseline, as a hardness check only:
    neither is one of the baselines the probe is compared with. Both use a logistic
    regression on the same folds.

    - ModernBERT-large: the mean of its last layer (`simulate/modernbert_frozen_app.py`).
    - bge-m3: its dense sentence embedding, computed locally and cached in
      `nb/data/.bge-m3_<kind>.npz` (the first run takes a while).
    """)
    return


@app.cell
def _(P, dslib, pd):
    encoders = pd.concat(
        {
            k: pd.DataFrame([dslib.frozen_encoder(k), dslib.m3_encoder(k)]).T
            for k in P.KINDS
        },
        axis=1,
    )
    encoders.loc[["plain", "natural", "adversarial", "ALL"]].round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Highest logistic-regression coefficients per expert, one table per vectorizer.
    """)
    return


@app.cell
def _(dslib, kind, pd):
    pd.concat(dslib.top_terms(kind.value), axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Adversarial pairs

    `pair_overlap`: Jaccard index of the word sets of the two samples in a pair
    (0 = no shared words, 1 = same words). Multi-turn uses the final developer turn only,
    because the earlier turns are shared. Single-turn uses the whole query.

    Higher overlap means a harder pair for bag-of-words. Pairs below
    `config.MIN_PAIR_OVERLAP` (0.60) are regenerated, so 0.60 is a floor.
    """)
    return


@app.cell
def _(dslib, plots):
    overlaps = dslib.pair_overlaps()
    print(overlaps.groupby("kind")["pair_overlap"].describe().round(3))
    plots.plot_pair_overlap(overlaps)
    return (overlaps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Summary
    """)
    return


@app.cell
def _(V, anova, ceiling, kind, overlaps, text_baselines):
    _gold = anova[anova.factor == "gold"].sort_values("omega2", ascending=False).iloc[0]
    _base = text_baselines[kind.value]
    print(f"kind                          {kind.value}")
    print(f"max Cramér's V, gold vs a non-structural factor  "
          f"{V.loc['gold', ['construction', 'persona', 'pattern']].max():.3f}")
    print(f"largest gold surface effect   {_gold.feature} omega2={_gold.omega2:.3f}")
    print(f"surface-only accuracy         {ceiling.loc['logreg, all features', 'ALL']:.3f} "
          f"(without pct_digit {ceiling.loc['logreg, without pct_digit', 'ALL']:.3f}, "
          f"without the numeric block {ceiling.loc['logreg, without numeric', 'ALL']:.3f}, "
          f"on adversarial {ceiling.loc['logreg, all features', 'adversarial']:.3f}, chance 0.200)")
    print(f"lexical / tfidf / embedding overall  {_base.loc['ALL', 'lexical']:.3f} / "
          f"{_base.loc['ALL', 'tfidf']:.3f} / {_base.loc['ALL', 'embedding']:.3f}")
    print(f"lexical on adversarial        {_base.loc['adversarial', 'lexical']:.3f}")
    print("mean pair overlap             " +
          ", ".join(f"{k} {v:.3f}" for k, v in overlaps.groupby('kind')['pair_overlap'].mean().items()))
    return


if __name__ == "__main__":
    app.run()
