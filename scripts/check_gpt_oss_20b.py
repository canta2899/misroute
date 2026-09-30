"""Smoke-test openai/gpt-oss-20b on modal"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--only-group simulation")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("simulate", "dataset")
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("check-gpt-oss-20b", image=image)

MODEL_KEY = "gpt-oss-20b"
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
    prompts = model.render(messages)
    print("\n===== prompt as rendered (tail) =====")
    print(prompts[0][-600:])

    routed = model.generate(messages)
    responses, hidden = routed.responses, routed.hidden
    print(f"\n[check] hidden states {hidden.shape} (layers, batch, d_model)")

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

    ok = _check_prompt_site(model, messages, routed) and ok

    hf_cache.commit()
    print("\n[check] ALL CHECKS PASSED" if ok else "\n[check] SOMETHING IS BROKEN")
    return ok


def _check_prompt_site(model, messages, routed) -> bool:
    """The prompt-site capture, unchanged when the same thing is run one
    sequence at a time rather than in a left-padded batch."""
    import torch

    ok = True
    print("\n[check] batched vs one-at-a-time (no padding), min cosine over layers:")
    for i, message in enumerate(messages):
        solo = model.generate([message])
        if solo.responses[0] != routed.responses[i]:
            print(f"  sample {i}: solo run generated different text, skipping")
            continue
        cos = torch.nn.functional.cosine_similarity(
            torch.from_numpy(routed.hidden[:, i, :]),
            torch.from_numpy(solo.hidden[:, 0, :]),
            dim=-1,
        )
        worst = float(cos.min())
        # bf16 through 24 layers is not bit-reproducible once a sequence is
        # padded, so exact agreement is only expected for the longest member of
        # a batch. A wrong probe site lands far below this, not just under 1.
        flag = "" if worst > 0.99 else "  <-- FAIL: wrong site, not fp noise"
        print(f"  sample {i} prompt: {worst:.6f}{flag}")
        if worst <= 0.99:
            ok = False
    return ok


@app.local_entrypoint()
def main():
    assert check.remote(), "tool-call parsing or hidden-state capture failed - see above"
