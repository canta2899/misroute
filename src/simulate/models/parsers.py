import json
import re


class TemplateParser:
    name = "base"
    CALL_RE: re.Pattern | None = None
    REASONING_RE: re.Pattern | None = None

    def __init__(
        self,
        tokenizer,
        *,
        processor=None,
        tools: list[dict] | None = None,
        system: str | None = None,
        template_kwargs: dict | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.tools = tools
        self.system = system
        self.template_kwargs = template_kwargs or {}
        self._prefix = None

    def __repr__(self) -> str:
        return self.name

    def conversation(self, messages: list[dict]) -> list[dict]:
        if not self.system:
            return list(messages)
        return [{"role": "system", "content": self.system}, *messages]

    def render(self, message_lists: list[list[dict]]) -> list[str]:
        return self.tokenizer.apply_chat_template(
            [self.conversation(m) for m in message_lists],
            tools=self.tools,
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )

    @property
    def generation_prefix(self) -> str:
        if self._prefix is None:
            self._prefix = self.render([[{"role": "user", "content": ""}]])[0]
        return self._prefix

    def tool_calls(self, text: str) -> list[tuple[str | None, dict]]:
        assert self.CALL_RE is not None
        calls: list[tuple[str | None, dict]] = []
        for match in self.CALL_RE.finditer(text):
            try:
                call = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            args = call.get("arguments") or {}
            calls.append((call.get("name"), args if isinstance(args, dict) else {}))
        return calls

    def strip_reasoning(self, text: str) -> str:
        if self.REASONING_RE is None:
            return text
        return self.REASONING_RE.sub("", text)


class HermesParser(TemplateParser):
    name = "hermes"
    CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    REASONING_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


class HarmonyParser(TemplateParser):
    name = "harmony"

    CALL_RE = re.compile(
        r"to=functions\.([A-Za-z_]\w*).*?<\|message\|>\s*(\{.*?\})\s*"
        r"(?:<\|call\|>|<\|end\|>|$)",
        re.DOTALL,
    )
    REASONING_RE = re.compile(
        r"<\|channel\|>analysis<\|message\|>.*?(?:<\|end\|>|$)", re.DOTALL
    )

    def tool_calls(self, text: str) -> list[tuple[str | None, dict]]:
        calls = []
        assert self.CALL_RE is not None
        for name, payload in self.CALL_RE.findall(text):
            try:
                args = json.loads(payload)
            except json.JSONDecodeError:
                args = {}
            calls.append((name, args if isinstance(args, dict) else {}))
        return calls


class Llama3Parser(TemplateParser):
    name = "llama3"
    CALL_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)

    def tool_calls(self, text: str) -> list[tuple[str | None, dict]]:
        assert self.CALL_RE is not None
        calls: list[tuple[str | None, dict]] = []
        for match in self.CALL_RE.finditer(text):
            try:
                call = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if not isinstance(call, dict) or "name" not in call:
                continue
            args = call.get("parameters") or call.get("arguments") or {}
            calls.append((call.get("name"), args if isinstance(args, dict) else {}))
        return calls


class Gemma4Parser(TemplateParser):
    name = "gemma4"

    def _parsed(self, text: str) -> dict:
        try:
            # gemma 4 ships a response_template in its tokenizer config
            return self.tokenizer.parse_response(text, prefix=self.generation_prefix)
        except ValueError:
            return {}

    def tool_calls(self, text: str) -> list[tuple[str | None, dict]]:
        return [
            (call["function"].get("name"), call["function"].get("arguments") or {})
            for call in self._parsed(text).get("tool_calls", [])
        ]

    def strip_reasoning(self, text: str) -> str:
        return self._parsed(text).get("content") or ""


PARSERS: dict[str, type[TemplateParser]] = {
    parser.name: parser
    for parser in (HermesParser, HarmonyParser, Llama3Parser, Gemma4Parser)
}


def get_parser(name: str) -> type[TemplateParser]:
    parser = PARSERS.get(name.lower(), None)
    if not parser:
        raise KeyError(f"unknown parser '{name}'. Available: {sorted(PARSERS)}")
    return parser
