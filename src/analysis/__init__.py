"""Analysis of the routing dataset and of the probes on the simulated runs.

Downstream of the pipeline: it reads the exported csvs and the merged parquets, and never
imports `dataset` or `simulate`. The marimo notebooks in `nb/` are its only callers.

    analysis.paths      where everything lives
    analysis.probe      loading a run, folds, the layer sweep and its cache
    analysis.dataset    surface features, balance, ANOVA, shortcut ceiling
    analysis.laya       laya (a decision model) against the coordinators and probes
    analysis.plots      every figure, shared by the notebooks and the report
"""
