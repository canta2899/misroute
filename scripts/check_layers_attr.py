"""Find the dotted path to the decoder layers ModuleList for each model (no weights downloaded)"""


import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch.nn as nn  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from simulate.models.models import MODELS  # noqa: E402


def find_layers_path(model: nn.Module) -> str | None:
    """Dotted path to the largest ModuleList in the model which is the decoder layers"""
    candidates = [(name, len(mod)) for name, mod in model.named_modules() if isinstance(mod, nn.ModuleList)]
    if not candidates:
        return None
    name, _ = max(candidates, key=lambda nc: nc[1])
    return name


def main() -> None:
    seen: dict[str, str] = {}  # model_name -> alias, to skip duplicate aliases
    for alias, model in MODELS.items():
        if model.model_name in seen:
            continue
        seen[model.model_name] = alias

        print(f"\n{alias} ({model.model_name})")
        try:
            config = AutoConfig.from_pretrained(model.model_name)
            with init_empty_weights():
                skeleton = AutoModelForCausalLM.from_config(config)
        except Exception as exc:
            print(f"  FAILED to build: {exc}")
            continue

        found = find_layers_path(skeleton)
        registered = model.layers_attr
        match = "OK" if found == registered else "MISMATCH"
        print(f"  registered layers_attr = {registered!r}")
        print(f"  found in architecture  = {found!r}")
        print(f"  -> {match}")


if __name__ == "__main__":
    main()
