import json
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Spec:
    system: str
    user: str
    tool: dict
    max_tokens: int


@dataclass
class Result:
    raw: dict
    output: dict | None = None
    error: str | None = None


@dataclass
class BatchStatus:
    done: bool
    counts: dict


class Provider(Protocol):
    model: str

    def call(self, spec: Spec) -> Result: ...
    def submit_batch(self, items: list[tuple[str, Spec]]) -> str: ...
    def poll_batch(self, batch_id: str) -> BatchStatus: ...
    def fetch_batch(self, batch_id: str) -> dict[str, Result]: ...


class AnthropicProvider:
    def __init__(self, model: str):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def _params(self, spec: Spec) -> dict:
        return {
            "model": self.model,
            "max_tokens": spec.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": spec.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": spec.user}],
            "tools": [spec.tool],
            "tool_choice": {"type": "tool", "name": spec.tool["name"]},
        }

    @staticmethod
    def _output(message) -> dict:
        for block in message.content:
            if block.type == "tool_use":
                return block.input
        raise ValueError(f"no tool_use block (stop_reason={message.stop_reason})")

    def call(self, spec: Spec) -> Result:
        message = self._client.messages.create(**self._params(spec))
        return Result(raw=message.to_dict(), output=self._output(message))

    def submit_batch(self, items) -> str:
        requests = [
            {"custom_id": cid, "params": self._params(spec)} for cid, spec in items
        ]
        return self._client.messages.batches.create(requests=requests).id  # type: ignore

    def poll_batch(self, batch_id: str) -> BatchStatus:
        batch = self._client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        return BatchStatus(
            done=batch.processing_status == "ended",
            counts={
                "processing": c.processing,
                "succeeded": c.succeeded,
                "errored": c.errored + c.canceled + c.expired,
            },
        )

    def fetch_batch(self, batch_id: str) -> dict:
        out = {}
        for entry in self._client.messages.batches.results(batch_id):
            if entry.result.type != "succeeded":
                out[entry.custom_id] = Result(
                    raw={"result_type": entry.result.type}, error=entry.result.type
                )
                continue
            message = entry.result.message
            try:
                out[entry.custom_id] = Result(message.to_dict(), self._output(message))
            except ValueError as exc:
                out[entry.custom_id] = Result(message.to_dict(), error=str(exc))
        return out


class OpenAIProvider:
    _DONE = {"completed", "failed", "expired", "cancelled"}

    def __init__(self, model: str):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI()

    def _params(self, spec: Spec) -> dict:
        return {
            "model": self.model,
            "max_completion_tokens": spec.max_tokens,
            "messages": [
                {"role": "system", "content": spec.system},
                {"role": "user", "content": spec.user},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": spec.tool["name"],
                        "description": spec.tool["description"],
                        "parameters": spec.tool["input_schema"],
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": spec.tool["name"]},
            },
        }

    @staticmethod
    def _output(message: dict) -> dict:
        calls = message.get("tool_calls") or []
        if not calls:
            raise ValueError("no tool call in response")
        return json.loads(calls[0]["function"]["arguments"])

    def call(self, spec: Spec) -> Result:
        raw = self._client.chat.completions.create(**self._params(spec)).model_dump()
        return Result(raw=raw, output=self._output(raw["choices"][0]["message"]))

    def submit_batch(self, items) -> str:
        lines = [
            json.dumps(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": self._params(spec),
                }
            )
            for cid, spec in items
        ]
        upload = self._client.files.create(
            file=("batch.jsonl", ("\n".join(lines) + "\n").encode()), purpose="batch"
        )
        return self._client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        ).id

    def poll_batch(self, batch_id: str) -> BatchStatus:
        batch = self._client.batches.retrieve(batch_id)
        c = batch.request_counts
        assert c is not None
        return BatchStatus(
            done=batch.status in self._DONE,
            counts={
                "processing": c.total - c.completed - c.failed,
                "succeeded": c.completed,
                "errored": c.failed,
            },
        )

    def fetch_batch(self, batch_id: str) -> dict:
        batch = self._client.batches.retrieve(batch_id)
        out = {}
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            for line in self._client.files.content(file_id).text.splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                out[entry["custom_id"]] = self._parse(entry)
        return out

    def _parse(self, entry: dict) -> Result:
        response = entry.get("response")
        if entry.get("error") or not response or response.get("status_code") != 200:
            return Result(raw=entry, error=str(entry.get("error") or response))
        try:
            return Result(
                entry, self._output(response["body"]["choices"][0]["message"])
            )
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return Result(raw=entry, error=str(exc))


_IMPLS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider}


def get_provider(model: str) -> Provider:
    name, _, model_id = model.partition("/")
    if not model_id or name not in _IMPLS:
        raise ValueError(
            f"model must be '<provider>/<model-id>' with provider in "
            f"{sorted(_IMPLS)}; got {model!r}"
        )
    return _IMPLS[name](model_id)
