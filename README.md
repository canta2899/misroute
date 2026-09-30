# Detecting Routing Failures in LLM-based Multi-Agent Systems through Linear Probes

This repository contains the code, data and LaTex source of my MSc thesis in Computer Science at the University of Udine, *Detecting Routing Failures in LLM-based Multi-Agent Systems through Linear Probes*.

It builds two datasets of developer requests whose correct sub-agent cannot be read off the wording, runs open-weight LLMs as coordinators on them, and trains linear probes on their hidden states to test whether a model encodes the right sub-agent even when it routes the request wrong.

The final datasets (single and multi turn) are in `dataset/`:

- `dataset/single_turn.csv`: one request per sample
- `dataset/multi_turn.csv`: a conversation per sample, routed at the last turn

The columns are described in [`dataset/README.md`](dataset/README.md).

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- an Anthropic API key, for dataset generation
- an OpenAI API key, only to generate with an `openai/...` model
- a [Modal](https://modal.com) account, for the simulation

## Setup

1. Install the dependencies:

   ```sh
   uv sync
   ```

2. Create `.env` in the repo root:

   ```sh
   ANTHROPIC_API_KEY=...
   OPENAI_API_KEY=...   # optional
   ```

3. Connect your Modal account:

   ```sh
   uv run modal setup
   ```

4. Create the Modal secret with your Hugging Face token. Llama 3.3 is a gated model and
   needs it:

   ```sh
   uv run modal secret create hf-token HF_TOKEN=...
   ```

   The volumes (`huggingface-cache`, `simulation-data`) are created on the first run.

## Dataset generation

Each run is a folder under `data/<run-id>/`, created by `ds plan`. Every other command
uses the most recent run, or the one given with `--run <id>`. Every command can run again:
it only does what is missing.

```sh
uv run ds plan [--name <id>] [--scale 1.0]   # decide every sample, no API calls
uv run ds facts                              # one constraint per scenario and expert
uv run ds generate --kind single|multi       # write the samples
uv run ds audit    --kind single|multi       # oracle: is the label recoverable
uv run ds analyze  --kind single|multi       # text baselines and checks, flags bad samples
uv run ds repair   --kind single|multi [--regenerate]   # delete flagged samples
uv run ds export                             # write dataset/<kind>.csv
uv run ds status                             # what exists, what is missing
```

- `--direct` calls the API one request at a time instead of the Batch API.
- `--limit N` stops after N requests (for a pilot).
- `--scale 0.02` on `plan` makes a small pilot run.

The scenarios every sample is built from are in `src/dataset/scenarios.json`.

### Review

```sh
uv run marimo run nb/review.py
```

The review notebook shows the samples of one run and kind, with the oracle verdict. You
can edit a sample by hand. It saves to `samples.edited.jsonl` and never changes
`samples.jsonl`. To export the edited file, run `uv run ds export --edited`.

## Simulation

Each model routes every sample on Modal. The run saves the model's response, its predicted
expert and the hidden states at the last prompt token.

### 1. Check the model

Run the check script for the model before the first simulation. It downloads the weights
into the `huggingface-cache` volume and checks that the tool call is parsed:

```sh
uv run modal run scripts/check_qwen3_8b.py
```

There is one `scripts/check_<model>.py` per model.

### 2. Run the simulation

Run from the repo root, in module mode (`-m`):

```sh
uv run modal run -m simulate.app --model-key qwen3-8b --dataset single_turn
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--model-key` | `qwen3-4b` | model to run (see `simulate.models.MODELS`) |
| `--dataset` | `single_turn` | `single_turn` or `multi_turn` |
| `--batch-size` | `32` | samples per forward pass |
| `--shard-size` | `50` | samples per output shard |
| `--limit` | none | run only the first N samples |
| `GPU_TYPE` (env) | `H100` | Modal GPU type |
| `NUM_GPU` (env) | `1` | number of GPUs; the layers are split across them |

Large models need more GPUs, and more GPUs allow a larger batch:

```sh
GPU_TYPE=H100 NUM_GPU=2 uv run modal run -m simulate.app --model-key llama3.3-70b --dataset multi_turn --batch-size 64
```

The shards go to the `simulation-data` volume, under `<model>/<kind>/shard_*.parquet`.

### 3. Download and merge the shards

```sh
uv run modal volume get simulation-data qwen3-8b data/sim/
uv run merge-shards --models qwen3-8b -o nb/data
```

`merge-shards` joins the shards with `dataset/<kind>.csv` and writes
`nb/data/<model>_<kind>.parquet`, the file the notebooks read. Without `--kind` it merges
both kinds.

## Notebooks

The notebooks read the merged parquets from `nb/data/`, one `<model>_<kind>.parquet` per
run. The parquets are not in the repository: run the simulation and `merge-shards` (see
above), or copy the merged files into `nb/data/` yourself.

```sh
uv run marimo edit nb/<notebook>.py
```

| Notebook | What it does |
| --- | --- |
| `nb/dataset.py` | checks that surface features (length, punctuation, digits) do not predict the label: design balance, ANOVA, a surface-only classifier, the text baselines, adversarial pair overlap |
| `nb/probing.py` | one run at a time: layer sweep, probe against the model and the text baselines. Then all runs: accuracy per tier, multi-turn drop, the probe as an error flag, depth of the best layer |
| `nb/failure.py` | compares the probe with the text baselines sample by sample: error dependence, adversarial pairs, decoy predictions |
| `nb/review.py` | hand review of the generated samples (see above) |

The layer sweeps are cached in `nb/data/cache/`, so the first run of `nb/probing.py` is slow
and later runs are fast. To fill the cache from the command line:

```sh
uv run python -m analysis.probe [model ...]
```

## Tests

```sh
uv run pytest
```

## Thesis

The thesis is in `thud/`, and all its figures are committed, so it builds without the data:

```sh
make -C thud
```

The figures of chapters 6-8 (`thud/images/models/`, `thud/images/plots/`) come from the
notebooks. To regenerate them, put the merged parquets in `nb/data/` and run:

```sh
make -C thud figures
```

## License

- Code: [Apache License 2.0](LICENSE)
- Dataset (`dataset/`): [CC BY 4.0](dataset/LICENSE)
- Thesis (`thud/`): [CC BY-NC-SA 4.0](thud/LICENSE)
