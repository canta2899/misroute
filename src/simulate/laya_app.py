"""Route the datasets with Laya (https://github.com/NandhaKishorM/laya), a
non-generative decision model: one forward pass scores a typed `choice`
question, so there is no text to parse and no hidden state to probe.

    uv run modal run -m simulate.laya_app
    uv run modal run -m simulate.laya_app --checkpoints english --limit 20
    uv run modal run -m simulate.laya_app --kinds multi_turn

Every run tries a grid of prompt variants, one container per (checkpoint, kind),
the checkpoint loaded once:

    options       how the five experts are written
                  handles    `<key>: <handles>`
                  tool       `<key>: <name> expert. Handles: <handles>`
                  tool_full  the coordinator's tool description verbatim, with the
                             example; laya cuts every option at 48 tokens, so the
                             example is cut (`options_cut`)
                  name       `<name>: <handles>`, no snake_case key
    instructions  the question text
                  task       one plain question
                  rule       the resolution rule in short form (binding vs waived)
                  system     the coordinator's system prompt verbatim
    state         what laya reads
                  turns      the `turns` json list (laya cuts it from the left)
                  text       the `text` column, `user: ... assistant: ...` (cut from the right)

The option and instruction texts are built locally from `simulate.experts`,
the same source as the coordinator's prompt, and passed to the container; they
are also written to <out-dir>/variants.json.

Budgets: the question and the options are never trimmed. `head_max_len` is the
checkpoint's own (192 / 256) unless a variant needs more, and `max_len` grows by
the same amount, so the room for the state stays the checkpoint's own
(512 / 1024 minus its head). Every single-turn sample fits it; multi-turn is
cut on `english` for most samples, see `over_budget`.

Self-contained in the container: the csvs are burned into the image, the
checkpoints are public (no HF token, no volume), and the results come back to
the local entrypoint, which writes one csv per kind to --out-dir.
"""

import csv
import json
import os
from pathlib import Path

import modal

REPO = "convaiinnovations/laya"
CHECKPOINTS = {  # name -> subfolder of the bundled hub repo
    "english": None,  # ModernBERT-large, trained at 512 tokens
    "multilingual": "multilingual",  # mmBERT-base, 1024 (up to 8192)
    "typed-decisions": "typed-decisions",  # ModernBERT-large, 1024
}
OPTIONS = ("handles", "tool", "tool_full", "name")
INSTRUCTIONS = ("task", "rule", "system")
STATES = ("turns", "text")

OPTION_MAX_TOKENS = 48  # laya's hard cap per option (laya.common.build_sequence)
BATCH_SIZE = 32

GPU = os.environ.get("GPU_TYPE", "L4")
DATASET_DIR = "/root/dataset"

TASK = "Which expert should handle this developer request?"
RULE = (
    "Exactly one requirement in this developer conversation is binding: "
    "someone with authority holds the team to it now. Every other "
    "requirement was waived by its owner. Which expert owns the binding one?"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("laya")
    .add_local_dir("dataset", DATASET_DIR, ignore=["merged"])
)

app = modal.App("laya-routing")


def build_variants(options, instructions, states) -> dict:
    """Every (options, instructions, state) as a laya question plus the label -> expert map.

    Runs locally: `simulate.experts` imports `dataset.config`, which the image does not carry.
    It is loaded by path, since script mode shadows the `simulate` package with simulate.py.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "laya_experts", Path(__file__).with_name("experts.py")
    )
    experts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experts)
    EXPERTS, SYSTEM_PROMPT = experts.EXPERTS, experts.SYSTEM_PROMPT

    # the tool description as simulate.models.base.BaseModel._build_tools writes it
    def tool_full(e):
        return (
            f"{e.name} expert. Handles: {e.handles}. "
            f'Example request that routes here: "{e.example}"'
        )

    option_sets = {
        "handles": {e.key: (e.key, e.handles) for e in EXPERTS},
        "tool": {
            e.key: (e.key, f"{e.name} expert. Handles: {e.handles}") for e in EXPERTS
        },
        "tool_full": {e.key: (e.key, tool_full(e)) for e in EXPERTS},
        "name": {e.key: (e.name, e.handles) for e in EXPERTS},
    }
    instruction_texts = {"task": TASK, "rule": RULE, "system": SYSTEM_PROMPT}

    variants = {}
    for o in options:
        for i in instructions:
            for s in states:
                opts = option_sets[o]
                variants[f"{o}|{i}|{s}"] = {
                    "options": o,
                    "instructions": i,
                    "state": s,
                    "labels": {label: key for key, (label, _) in opts.items()},
                    "question": {
                        "type": "choice",
                        "instructions": instruction_texts[i],
                        "criteria": {label: text for label, text in opts.values()},
                    },
                }
    return variants


def budgets(tok, cfg: dict, q: dict) -> dict:
    """`max_len` / `head_max_len` for one question, the state's `room` and `options_cut`.

    The question and every option are whole, never less than the checkpoint's own head;
    the state keeps the room the checkpoint was trained with.
    """
    n_tokens = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    default_max_len = cfg.get("max_len", 512)
    default_head = cfg.get("head_max_len", 192)
    head = n_tokens(f"choice question: {q['instructions']}")
    opt_full = [n_tokens(f" {k}: {d}") for k, d in q["criteria"].items()]
    opts = sum(1 + min(n, OPTION_MAX_TOKENS) for n in opt_full)
    head_max_len = max(default_head, opts + max(head, 16))
    max_len = default_max_len + head_max_len - default_head
    return {
        "max_len": max_len,
        "head_max_len": head_max_len,
        # [CLS] head [SEP] options [SEP] state [SEP]
        "room": max_len - (head + opts + 3) - 1,
        "options_cut": max(opt_full) > OPTION_MAX_TOKENS,
    }


def result_row(
    sample: dict, model: str, name: str, v: dict, ans: dict, n_state: int, b: dict
) -> dict:
    """One csv row: a sample routed by one checkpoint with one variant."""
    predicted = v["labels"][ans["choice"]]
    return {
        "id": sample["id"],
        "kind": sample["kind"],
        "model": model,
        "variant": name,
        "options": v["options"],
        "instructions": v["instructions"],
        "state": v["state"],
        "gold": sample["gold"],
        "predicted": predicted,
        "correct": predicted == sample["gold"],
        "answer_confidence": ans["answer_confidence"],
        **{
            f"p_{key}": ans["probabilities"][label]
            for label, key in v["labels"].items()
        },
        "n_state_tokens": n_state,
        "over_budget": n_state > b["room"],
        "options_cut": b["options_cut"],
        "max_len": b["max_len"],
        "head_max_len": b["head_max_len"],
    }


@app.function(image=image, gpu=GPU, timeout=60 * 60)
def run_laya(
    checkpoint: str,
    kind: str,
    variants: dict,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
) -> list[dict]:
    import laya
    from laya.common import serialize_state

    with open(f"{DATASET_DIR}/{kind}.csv", newline="") as f:
        samples = list(csv.DictReader(f))[:limit]

    agent = laya.load(REPO, subfolder=CHECKPOINTS[checkpoint], device="cuda")
    n_tokens = lambda s: len(agent.tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731

    # a conversation list is left-truncated by laya, so the newest turn survives;
    # a plain string is cut from the right
    states = {
        "turns": [json.loads(s["turns"]) for s in samples],
        "text": [s["text"] for s in samples],
    }
    state_tokens = {
        name: [n_tokens(serialize_state(x)) for x in xs] for name, xs in states.items()
    }

    rows = []
    for name, v in variants.items():
        b = budgets(agent.tok, agent.cfg, v["question"])
        results = agent.predict_batch(
            states[v["state"]],
            {"expert": v["question"]},
            batch_size=batch_size,
            max_len=b["max_len"],
            head_max_len=b["head_max_len"],
            sort_by_length=True,
        )
        new = [
            result_row(
                sample, f"laya-{checkpoint}", name, v, res["answers"]["expert"], n, b
            )
            for sample, n, res in zip(samples, state_tokens[v["state"]], results)
        ]
        rows += new
        print(
            f"[laya] {checkpoint} {kind} {name}: max_len={b['max_len']} "
            f"head_max_len={b['head_max_len']} "
            f"acc={sum(r['correct'] for r in new) / max(len(new), 1):.3f}"
        )
    return rows


@app.local_entrypoint()
def main(
    checkpoints: str = ",".join(CHECKPOINTS),
    kinds: str = "single_turn",
    options: str = ",".join(OPTIONS),
    instructions: str = ",".join(INSTRUCTIONS),
    states: str = ",".join(STATES),
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    out_dir: str = "data/sim/laya",
):
    variants = build_variants(
        options.split(","), instructions.split(","), states.split(",")
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "variants.json").write_text(json.dumps(variants, indent=2))

    jobs = [
        (c, k, variants, batch_size, limit)
        for c in checkpoints.split(",")
        for k in kinds.split(",")
    ]
    rows = [r for batch in run_laya.starmap(jobs) for r in batch]

    for kind in kinds.split(","):
        kind_rows = [r for r in rows if r["kind"] == kind]
        path = out / f"{kind}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(kind_rows[0]))
            writer.writeheader()
            writer.writerows(kind_rows)
        print(f"[laya] wrote {len(kind_rows)} rows -> {path}")
