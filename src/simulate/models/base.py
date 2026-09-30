from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from simulate.experts import EXPERT_KEYS, EXPERTS, SYSTEM_PROMPT

from .parsers import HermesParser, TemplateParser


class Routed(NamedTuple):
    """One batch routed, with the hidden states captured for it."""

    responses: list[str]
    hidden: np.ndarray  # [n_layers, batch, hidden_dim]


class BaseModel:
    def __init__(
        self,
        model_name: str,
        *,
        parser: type[TemplateParser] = HermesParser,
        template_kwargs: dict | None = None,
        gen_kwargs: dict | None = None,
        max_new_tokens: int = 128,
        layers_attr: str = "model.layers",
        system_extra: dict[str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.parser_cls = parser
        self.template_kwargs = template_kwargs or {}
        self.system_extra = system_extra or {}
        self.gen_kwargs = gen_kwargs or {"do_sample": False}
        self.max_new_tokens = max_new_tokens
        self.layers_attr = layers_attr

        self.tokenizer = None
        self.processor = None
        self.model = None
        self.parser: TemplateParser | None = None
        self.tools = self._build_tools()

        self._captured = {}  # layer_idx -> last prompt-token hidden state (cpu)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.model_name}, parser={self.parser_cls.name})"
        )

    def _build_tools(self) -> list[dict]:
        """The five experts as tool definitions, one per route."""
        return [
            {
                "type": "function",
                "function": {
                    "name": e.key,
                    "description": (
                        f"{e.name} expert. Handles: {e.handles}. "
                        f'Example request that routes here: "{e.example}"'
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            for e in EXPERTS
        ]

    def system_prompt(self, kind: str | None = None) -> str:
        """The shared prompt, plus this model's line for `kind` if it has one."""
        extra = self.system_extra.get(kind or "", "")
        return f"{SYSTEM_PROMPT}\n\n{extra}" if extra else SYSTEM_PROMPT

    def load_parser(self, kind: str | None = None) -> "BaseModel":
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            # some models may have it and it can be used to process the output
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception as exc:
            print(f"[load] no processor for {self.model_name}: {exc}")
            self.processor = None

        self.parser = self.parser_cls(
            self.tokenizer,
            processor=self.processor,
            tools=self.tools,
            system=self.system_prompt(kind),
            template_kwargs=self.template_kwargs,
        )
        return self

    def load(self, *, kind: str | None = None) -> "BaseModel":
        """Load weights + tokenizer and register the hidden-state hooks."""
        self.load_parser(kind)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", dtype="auto"
        )
        self.model.eval()

        for i, layer in enumerate(self.layer_modules()):
            layer.register_forward_hook(self._make_hook(i))

        return self

    def layer_modules(self) -> torch.nn.ModuleList:
        """The decoder layers to hook, from `layers_attr`."""
        module = self.model
        for part in self.layers_attr.split("."):
            module = getattr(module, part)
        return module  # pyright: ignore[reportReturnType]

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module, inp, output):  # pyright: ignore[reportUnusedParameter]
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] > 1:  # the prefill: its last token is the prompt site
                self._captured[layer_idx] = hidden[:, -1, :].to(
                    "cpu", dtype=torch.float32
                )

        return hook

    def render(self, message_lists: list[list[dict]]) -> list[str]:
        assert self.parser is not None, "call load() or load_parser() first"
        return self.parser.render(message_lists)

    def generate(
        self, message_lists: list[list[dict]], max_new_tokens: int | None = None
    ) -> Routed:
        """Route a batch, capturing the hidden states the probe reads."""
        assert self.tokenizer is not None
        assert self.model is not None

        self._captured.clear()

        # the chat template emits bos itself, hence add_special_tokens=False
        inputs = self.tokenizer(
            self.render(message_lists),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                return_dict_in_generate=True,
                **self.gen_kwargs,
            )

        # we don't skip special token because we need them to parse the tool call
        responses = self.tokenizer.batch_decode(
            out.sequences[:, prompt_len:], skip_special_tokens=False
        )
        hidden = torch.stack([self._captured[i] for i in sorted(self._captured)], dim=0)
        return Routed(responses, hidden.numpy())

    def parse(self, text: str) -> str | None:
        """The expert this response routes to."""
        assert self.parser is not None, "call load() or load_parser() first"

        calls = self.parser.tool_calls(text)
        if not calls:
            return None

        name, _ = calls[0]
        return name if name in EXPERT_KEYS else None
