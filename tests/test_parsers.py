"""Each parser against the exact call format its own chat template renders.

The strings below were taken from the templates themselves (rendering an
assistant message with `tool_calls` through `apply_chat_template`), not written
by hand.
"""

import pytest

from simulate.models import get_model


def model(key):
    """The registry instance with its parser built, but no weights.

    A parser talks through the tokenizer - gemma 4 parses with the tokenizer's
    own `response_template` - so there is nothing to test without it.
    """
    m = get_model(key)
    if m.parser is None:
        try:
            m.load_parser()
        except Exception as exc:  # offline / gated
            pytest.skip(f"{m.model_name}: tokenizer unavailable: {exc}")
    return m

CALLS = {
    "qwen3-8b": (
        "<tool_call>\n"
        '{"name": "time_to_market", "arguments": {}}\n'
        "</tool_call>"
    ),
    "gpt-oss-20b": (
        "<|start|>assistant to=functions.time_to_market<|channel|>commentary json"
        "<|message|>{}<|call|>"
    ),
    "gemma4-12b-it": "<|tool_call>call:time_to_market{}<tool_call|>",
    "llama3.3-70b": '{"name": "time_to_market", "parameters": {}}',
}

TRACES = {
    "qwen3-8b": "<think>\nscale_arch is tempting but the demo date binds\n</think>\n",
    "gpt-oss-20b": "<|channel|>analysis<|message|>scale_arch? no<|end|>",
    "gemma4-12b-it": "<|channel>thought\nnot scale_arch<channel|>",
    # llama has no trace format; plain prose is what it writes instead
    "llama3.3-70b": "scale_arch is tempting but the demo date binds.\n",
}


@pytest.mark.parametrize("model_key", sorted(CALLS))
def test_parses_call(model_key):
    assert model(model_key).parse(CALLS[model_key]) == "time_to_market"


@pytest.mark.parametrize("model_key", sorted(CALLS))
def test_parses_call_after_a_thinking_trace(model_key):
    text = TRACES[model_key] + CALLS[model_key]
    assert model(model_key).parse(text) == "time_to_market"


@pytest.mark.parametrize("model_key", sorted(CALLS))
def test_no_call_is_unrouted(model_key):
    assert model(model_key).parse("i need more detail before routing.") is None


@pytest.mark.parametrize("model_key", sorted(CALLS))
def test_expert_named_only_while_thinking_is_not_a_route(model_key):
    """A model weighing scale_arch out loud and then not committing.

    The fallback reads the answer, not the trace.
    """
    assert model(model_key).parse(TRACES[model_key]) is None
