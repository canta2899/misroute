"""Smoke-test google/gemma-4-E2B-it on modal

`google/gemma-4-E2B` is the base checkpoint and ships no chat template, so the
instruction-tuned one is what gets routed here.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--only-group simulation")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("simulate", "dataset")
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("check-gemma4-e2b", image=image)

MODEL_KEY = "gemma4-e2b-it"
GPU = "H100"

PROMPTS = [
    "we're demoing the ingest flow to the buyer on thursday and the deal is "
    "contingent on it running end to end. i'd rather wire the mapping properly "
    "but that can wait.",
    "two external customers still call the v1 export endpoint and their contracts "
    "say we can't change its response shape for another year. i want to rework "
    "what happens behind it, ideally making it faster too.",
]


@app.function(gpu=GPU, volumes={"/cache": hf_cache}, timeout=60 * 60)
def check():
    import os

    os.environ.setdefault("HF_HOME", "/cache")

    from simulate.models import get_model

    model = get_model(MODEL_KEY)
    print(f"[check] {MODEL_KEY} -> {model!r}")
    model.load()
    print(f"[check] loaded, layers={len(model.layer_modules())}, "
          f"max_new_tokens={model.max_new_tokens}, tkw={model.template_kwargs}, "
          f"gen={model.gen_kwargs}")

    messages = [[{"role": "user", "content": p}] for p in PROMPTS]
    print("\n===== prompt as rendered (tail) =====")
    print(model.render(messages)[0][-600:])

    routed = model.generate(messages)
    responses, hidden = routed.responses, routed.hidden
    print(f"\n[check] hidden states {hidden.shape} (layers, batch, d_model)")
    if hidden.shape[0] != len(model.layer_modules()):
        print("FAIL: one vector per decoder layer expected")

    ok = True
    for prompt, response in zip(PROMPTS, responses):
        expert = model.parse(response)
        print("\n===== raw generation =====")
        print(repr(response))
        print(f"query    : {prompt[:60]}...")
        print(f"parsed   : expert={expert!r}")
        if expert is None:
            ok = False
            print("FAIL: nothing routed")

    hf_cache.commit()
    print("\n[check] TOOL CALLS READ CORRECTLY" if ok else "\n[check] PARSING BROKEN")
    return ok


@app.local_entrypoint()
def main():
    assert check.remote(), "tool-call parsing failed - see the raw generations above"
