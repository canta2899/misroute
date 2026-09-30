# MisRoute

A synthetic dataset for coordinator routing, and hidden-state probing on top of it.

- `src/dataset/` generates two datasets (single-turn requests and multi-turn
  conversations). For each developer request, the dataset gives the sub-agent (expert)
  a coordinator should route it to.
- `src/simulate/` runs open-weight LLMs as coordinators on Modal. It records the
  predicted expert and the hidden states at the last prompt token.
- `src/analysis/` trains linear probes per layer on those hidden states. It compares
  them against the model's own choice and against text-only baselines (tf-idf and
  embedding logistic regression).

The question is whether a hidden layer encodes the right expert more accurately than
the model does. That only means something if the label cannot be read off the request
text, so the datasets must stay hard for the text-only baselines.

Commands and setup are in `README.md`.

## Layout

- `src/dataset/scenarios.json`: the seed pool of technical situations every sample is built from
- `dataset/<kind>.csv`: the exported datasets, the input of simulate (`<kind>.meta.json` is gitignored)
- `data/<run-id>/`: generation runs (gitignored)
- `nb/`: marimo notebooks (`dataset`, `probing`, `failure`, `laya`, `review`)
- `nb/data/`: merged parquets and layer-sweep caches (gitignored)
- `scripts/check_<model>.py`: per-model smoke tests on Modal
- `tests/`: unit tests (`uv run pytest`)

## Dataset generation

The taxonomy is **constraint-based**. The five experts are not topics: they look at the
same system and optimise for different constraints (`time_to_market`, `scale_arch`,
`resource_opt`, `legacy_compat`, `strict_compliance`). The label is the binding
constraint, decided by `config.RESOLUTION_RULE`: exactly one constraint is a mandate,
and everything else is a preference.

Pipeline stages, one `ds` subcommand each:

    plan      decide every sample offline: deterministic, balanced, no API calls
    facts     one operating constraint per (scenario, expert)
    generate  render plan rows into queries / conversations, filling plan gaps
    audit     route each sample with the rubric (oracle) -> oracle_* fields
    analyze   baselines and checks; annotates samples, writes read-only reports
    repair    delete samples marked `regenerate` so generate recreates them
    export    write dataset/<kind>.csv + <kind>.meta.json

Modules: `config` (taxonomy and knobs), `plan`, `prompts`, `providers` + `llm` (model
access), `generate`, `checks` (structural rules), `analyze`, `audit`, `repair`, `store`
(paths and json lines), `reviewlib` (the review pass without a UI; `nb/review.py` is
the front end), `csvexport` (flattening samples to csv).

Design rules to preserve:
- the plan is the source of truth for *what should exist*. Generation only fills gaps,
  so every stage can be re-run, and pilots are just `--scale` / `--limit`
- `samples.jsonl` is the source of truth for *the state of what exists*: the text, the
  oracle verdict (`oracle_*`), every check (`structural_ok`, `duplicate_of`,
  `duplicate_similarity`, `pair_overlap`), the human's call (`review_decision`,
  `review_note`) and the `regenerate` switch that repair acts on
- the review notebook writes `samples.edited.jsonl`, never `samples.jsonl`
- files under `reports/` are summaries for a human. Nothing in the pipeline reads them back
- adversarial samples are contrastive minimal pairs, generated in one call and always
  regenerated as a unit
- `analyze` annotates samples with check results. Only generation and repair change the text
- splits and CV folds are grouped by scenario, never by sample

## Simulation

Run from the repo root in module mode (`uv run modal run -m simulate.app`). Script mode
shadows the `simulate` package with `simulate/simulate.py`. Simulate reads
`dataset/<kind>.csv` (the `turns` column is the prompt; `text` is for reading only) and
writes `<model>/<kind>/shard_*.parquet`. `merge-shards` joins the shards with the csv
into `nb/data/<model>_<kind>.parquet`, the file the notebooks read.

## Analysis

    src/analysis/paths.py    where everything lives, the only module that knows the layout
    src/analysis/probe.py    loading a run, folds, layer sweep + cache
    src/analysis/dataset.py  surface features, balance, ANOVA, shortcut ceiling
    src/analysis/laya.py     laya (non-generative decision model) against every router
    src/analysis/plots.py    every figure

- `analysis` is downstream of the pipeline: it reads `dataset/*.csv` and the merged
  parquets and never imports `dataset` or `simulate`
- the notebooks are the source of truth for every result. New analysis goes in
  `src/analysis/` and gets a notebook cell. Any script that renders figures or prints
  numbers only calls `analysis` functions and holds no analysis step of its own
- choices shared by a notebook and a script (which runs to pool, which layer) live in
  `analysis` as a named function, never duplicated
