"""Smoke-test meta-llama/Llama-3.3-70B-Instruct on modal"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--only-group simulation")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("simulate", "dataset")
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("check-llama33-70b", image=image)

MODEL_KEY = "llama3.3-70b"
GPU = "H100:4"

PROMPTS = [
    "we're demoing the ingest flow to the buyer on thursday and the deal is "
    "contingent on it running end to end. i'd rather wire the mapping properly "
    "but that can wait.",
    "two external customers still call the v1 export endpoint and their contracts "
    "say we can't change its response shape for another year. i want to rework "
    "what happens behind it, ideally making it faster too.",
]


@app.function(
    gpu=GPU,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=60 * 60,
)
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
    prompt = model.render(messages)[0]
    print("\n===== prompt as rendered (tail) =====")
    print(prompt[-900:])

    ok = True
    # a template that drops `tools` renders a prompt with no expert in it, and
    # every route after that is the model guessing at names it never saw
    if "time_to_market" not in prompt:
        ok = False
        print("FAIL: the template rendered no tools - the prompt names no expert")

    routed = model.generate(messages)
    responses, hidden = routed.responses, routed.hidden
    print(f"\n[check] hidden states {hidden.shape} (layers, batch, d_model)")

    for query, response in zip(PROMPTS, responses):
        expert = model.parse(response)
        print("\n===== raw generation =====")
        print(repr(response))
        print(f"query    : {query[:60]}...")
        print(f"parsed   : expert={expert!r}")
        if expert is None:
            ok = False
            print("FAIL: nothing routed")

    hf_cache.commit()
    print("\n[check] TOOL CALLS READ CORRECTLY" if ok else "\n[check] PARSING BROKEN")
    return ok


@app.local_entrypoint()
def main():
    assert check.remote(), "see the rendered prompt and generations above"
