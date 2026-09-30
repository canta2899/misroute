import os

import modal

from simulate.modal_image import hf_cache, image, sim_data

image = (
    image.env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("simulate", "dataset")
    .add_local_dir("dataset", "/root/csv")  # the exported csvs
)

BATCH_SIZE = 32

GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
NUM_GPU = int(os.environ.get("NUM_GPU", "1"))
GPU = f"{GPU_TYPE}:{NUM_GPU}"

HF_CACHE_DIR = "/cache"
DATA_DIR = "/data"
DATASET_DIR = "/root/csv"

app = modal.App("routing-simulation")


@app.function(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE_DIR: hf_cache, DATA_DIR: sim_data},
    timeout=6 * 60 * 60,
)
def simulate_on_modal(
    model_key: str = "qwen3-4b",
    dataset: str = "single_turn",
    batch_size: int = BATCH_SIZE,
    shard_size: int = 50,
    limit: int | None = None,
):
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

    from simulate.models import get_model
    from simulate.simulate import run_simulation

    model = get_model(model_key).load(kind=dataset)
    run_simulation(
        model,
        f"{DATASET_DIR}/{dataset}.csv",
        f"{DATA_DIR}/{model_key}/{dataset}",
        batch_size=batch_size,
        shard_size=shard_size,
        limit=limit,
        on_shard=lambda _path: sim_data.commit(),
    )
    hf_cache.commit()


@app.local_entrypoint()
def main(
    model_key: str = "qwen3-4b",
    dataset: str = "single_turn",
    batch_size: int = BATCH_SIZE,
    shard_size: int = 50,
    limit: int | None = None,
):
    simulate_on_modal.remote(
        model_key=model_key,
        dataset=dataset,
        batch_size=batch_size,
        shard_size=shard_size,
        limit=limit,
    )
