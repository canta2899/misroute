from simulate.models.base import BaseModel
from simulate.models.parsers import (
    Gemma4Parser,
    HarmonyParser,
    HermesParser,
    Llama3Parser,
)

MODELS: dict[str, BaseModel] = {}


def register(model: BaseModel, *aliases: str) -> BaseModel:
    for alias in aliases:
        MODELS[alias] = model
    return model


QWEN3_4B = register(
    BaseModel(
        "Qwen/Qwen3-4B-Instruct-2507",
        parser=HermesParser,
        gen_kwargs={"do_sample": False},
        max_new_tokens=128,
    ),
    "qwen3-4b",
    "qwen3-4b-it",
)

QWEN25_72B = register(
    BaseModel(
        "Qwen/Qwen2.5-72B-Instruct",
        parser=HermesParser,
        gen_kwargs={"do_sample": False},
        max_new_tokens=128,
    ),
    "qwen2.5-72b",
    "qwen2.5-72b-it",
)

QWEN3_8B = register(
    BaseModel(
        "Qwen/Qwen3-8B",
        parser=HermesParser,
        template_kwargs={"enable_thinking": False},
        gen_kwargs={"do_sample": False},
        max_new_tokens=128,
    ),
    "qwen3-8b",
)

GPT_OSS_20B = register(
    BaseModel(
        "openai/gpt-oss-20b",
        parser=HarmonyParser,
        template_kwargs={"reasoning_effort": "low"},
        gen_kwargs={"do_sample": False},
        max_new_tokens=512,
    ),
    "gpt-oss-20b",
)

GEMMA4_12B = register(
    BaseModel(
        "google/gemma-4-12B-it",
        parser=Gemma4Parser,
        template_kwargs={"enable_thinking": False},
        gen_kwargs={"do_sample": False},
        # no analysis block to leave room for: the tool call is all it writes
        max_new_tokens=128,
        layers_attr="model.language_model.layers",
    ),
    "gemma4-12b-it",
    "gemma4-12B-it",
)

LLAMA33_70B = register(
    BaseModel(
        "meta-llama/Llama-3.3-70B-Instruct",
        parser=Llama3Parser,
        gen_kwargs={"do_sample": False},
        max_new_tokens=128,
    ),
    "llama3.3-70b",
)

GEMMA4_26B_A4B = register(
    BaseModel(
        "google/gemma-4-26B-A4B-it",
        parser=Gemma4Parser,
        template_kwargs={"enable_thinking": False},
        gen_kwargs={"do_sample": False},
        max_new_tokens=128,
        layers_attr="model.language_model.layers",
    ),
    "gemma4-26b-a4b-it",
    "gemma4-26B-A4B-it",
)

GEMMA4_E2B = register(
    BaseModel(
        "google/gemma-4-E2B-it",
        parser=Gemma4Parser,
        template_kwargs={"enable_thinking": False},
        gen_kwargs={"do_sample": False},
        max_new_tokens=128,
        layers_attr="model.language_model.layers",
        # the small one answers multi-turn conversations in prose and leaves the
        # sample unrouted; single-turn needs no nudge
        system_extra={"multi_turn": "ALWAYS answer with exactly one tool call."},
    ),
    "gemma4-e2b-it",
    "gemma4-E2B-it",
)


def get_model(name: str) -> BaseModel:
    model = MODELS.get(name, None)
    if not model:
        raise KeyError(f"unknown model '{name}'")
    return model
