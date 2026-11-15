"""A stub Tinker service for tests: the same method names as the real SDK, no network.

`FakeService` mirrors `tinker.ServiceClient`: `create_lora_training_client` returns a
future of a `FakeTrainingClient` with `forward_backward`, `optim_step`,
`save_weights_and_get_sampling_client` and `get_tokenizer`. `FakeTypes` mirrors
`tinker.types` (`ModelInput.from_ints`, `Datum`, `AdamParams`).

Scoring is deterministic: a token's logprob is high if it ever appeared as a weighted
target in training, plus a small token-dependent jitter so argmax over verbalizers is
stable but not constant. That yields a plausible (not all-0/all-1) R matrix end-to-end.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


class FakeFuture:
    def __init__(self, value: Any):
        self._value = value

    def result(self, timeout: float | None = None) -> Any:
        return self._value

    def __await__(self) -> Iterator[Any]:
        yield self._value


class FakeModelInput:
    def __init__(self, tokens: list[int]):
        self.tokens = list(tokens)

    @classmethod
    def from_ints(cls, tokens: list[int]) -> FakeModelInput:
        return cls(tokens)


@dataclass
class FakeDatum:
    model_input: FakeModelInput
    loss_fn_inputs: dict[str, Any]


@dataclass
class FakeAdamParams:
    learning_rate: float


class FakeTypes:
    ModelInput = FakeModelInput
    Datum = FakeDatum
    AdamParams = FakeAdamParams


class FakeTokenizer:
    """Deterministic word-level tokenizer: one id per whitespace-separated token."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [zlib.crc32(w.encode()) % 60000 + 10 for w in text.split()]


class FakeTrainingClient:
    def __init__(self, service: FakeService, base_model: str, rank: int, seed: int):
        self._service = service
        self.base_model = base_model
        self.rank = rank
        self.seed = seed
        self.received: list[FakeDatum] = []
        self.optim_steps = 0

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()

    def forward_backward(self, data, loss_fn: str) -> FakeFuture:
        for d in data:
            self.received.append(d)
            weights = d.loss_fn_inputs["weights"]
            targets = d.loss_fn_inputs["target_tokens"]
            for w, t in zip(weights, targets):
                if w:
                    self._service.trained_tokens.add(t)
            self._service.train_tokens += len(d.model_input.tokens) + 1
        return FakeFuture({"metrics": {"loss:sum": 1.0}})

    def optim_step(self, params) -> FakeFuture:
        self.optim_steps += 1
        self._service.last_lr = getattr(params, "learning_rate", None)
        return FakeFuture({})

    def save_weights_and_get_sampling_client(self) -> FakeFuture:
        return FakeFuture(FakeSamplingClient(self._service))


class FakeSamplingClient:
    def __init__(self, service: FakeService):
        self._service = service

    def _logprob(self, token: int) -> float:
        base = -0.5 if token in self._service.trained_tokens else -2.0
        return base - 0.01 * (token % 13)

    def compute_logprobs(self, prompt, prompt_logprobs_last_n: int | None = None) -> FakeFuture:
        toks = prompt.tokens
        n = len(toks) if prompt_logprobs_last_n is None else prompt_logprobs_last_n
        self._service.prefill_tokens += len(toks)
        lps: list[float | None] = [None] * len(toks)
        for i in range(len(toks) - n, len(toks)):
            if i > 0:
                lps[i] = self._logprob(toks[i])
        return FakeFuture(lps)


class FakeService:
    """Drop-in for tinker.ServiceClient. Tracks token totals for the tests."""

    def __init__(self):
        self.trained_tokens: set[int] = set()
        self.train_tokens = 0
        self.prefill_tokens = 0
        self.last_lr: float | None = None
        self.clients: list[FakeTrainingClient] = []

    def create_lora_training_client(self, base_model: str, rank: int = 32,
                                    seed: int | None = None, **kw) -> FakeFuture:
        client = FakeTrainingClient(self, base_model, rank, seed or 0)
        self.clients.append(client)
        return FakeFuture(client)
