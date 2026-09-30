"""
Sanity check for the embedding model
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from dataset import config  # noqa: E402
from dataset.analyze import embed as embed_texts  # noqa: E402

SENTENCES = [
    "my postgres query got slower as the table grew",
    "the postgres query takes much longer now that the table is bigger",
    "write a contributing guide for external contributors",
]


def main() -> None:
    print(f"embedding model: {config.EMBEDDING_MODEL}")
    print("loading (downloads from Hugging Face on first run) ...")

    vectors = embed_texts(SENTENCES)

    assert vectors.shape[0] == len(SENTENCES), f"unexpected row count: {vectors.shape}"
    assert vectors.ndim == 2 and vectors.shape[1] > 0, (
        f"unexpected shape: {vectors.shape}"
    )
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"vectors not normalized: {norms}"

    sim_paraphrase = float(vectors[0] @ vectors[1])
    sim_unrelated = float(vectors[0] @ vectors[2])
    print(f"shape: {vectors.shape}")
    print(f"cosine(paraphrase pair) = {sim_paraphrase:.4f}")
    print(f"cosine(unrelated pair)  = {sim_unrelated:.4f}")
    assert sim_paraphrase > sim_unrelated, (
        "similarity sanity check failed: paraphrases should be closer than unrelated text"
    )
    assert (
        sim_paraphrase > config.DEDUP_COSINE_THRESHOLD * 0.9 or sim_paraphrase > 0.8
    ), f"paraphrase similarity {sim_paraphrase:.3f} suspiciously low"

    print(
        f"OK — {config.EMBEDDING_MODEL} is downloaded, normalized, and sane "
        f"(dedup threshold in config: {config.DEDUP_COSINE_THRESHOLD})."
    )


if __name__ == "__main__":
    main()
