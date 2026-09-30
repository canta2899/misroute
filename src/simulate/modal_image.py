import modal

image = modal.Image.debian_slim(python_version="3.12").uv_sync(
    extra_options="--only-group simulation"
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
sim_data = modal.Volume.from_name("simulation-data", create_if_missing=True)
