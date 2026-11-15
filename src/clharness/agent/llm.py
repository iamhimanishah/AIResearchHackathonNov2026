"""Model access for the four model agents: one protocol, one fake, one real client, one guard.

  * `LLMClient`: `complete(agent, system, user, schema) -> (validated object, Usage)`.
  * `FakeLLM`: scripted answers for tests. Records every full prompt so a test can scan them.
  * `OpenAIClient`: the real client. The SDK is imported lazily. UNVERIFIED against the live API:
    it has only been exercised with an injected fake SDK object. Check it by hand before the grid.
  * `BudgetedLLM`: wraps any client. Checks the model-dollar cap BEFORE each call, retries once on
    invalid output, prices the call, and appends one line per attempt to `llm_calls.jsonl`.

Agents always talk to a `BudgetedLLM`. Raw clients report tokens; the wrapper turns them into
dollars with the configured prices, so the cap holds for any client.

No network client is imported at module import time.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from .config import AgentConfig, ModelPrice
from .contracts import Frozen, append_jsonl, read_jsonl, sha256_hex, utc_iso

T = TypeVar("T", bound=BaseModel)
FAKE_MODEL = "fake"
GARBAGE = "<<this is not json>>"  # script a FakeLLM with this to get invalid output


class LLMError(Exception):
    """Base of every model-call failure. The Conductor falls back when it sees one."""


class InvalidOutput(LLMError):
    """The model answered, but not with an object the schema accepts. Tokens were still spent."""

    def __init__(self, message: str, raw: str = "", usage: Usage | None = None) -> None:
        super().__init__(message)
        self.raw = raw
        self.usage = usage


class BudgetExceeded(LLMError):
    """The call was refused before it was made: it could take spending past the cap."""


class ModelNotConfigured(LLMError):
    """No model name, no price or no SDK for this agent."""


class LLMUnavailable(LLMError):
    """The provider failed (network, rate limit, server error)."""


class Usage(Frozen):
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float = 0.0  # filled by BudgetedLLM
    prompt_hash: str = ""  # hash of the prompt of the attempt that succeeded
    attempts: int = 1


class LLMCallRecord(Frozen):
    """One line of `llm_calls.jsonl`."""

    ts: str
    agent: str
    prompt_hash: str
    model: str
    schema_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ok: bool
    attempt: int
    error: str = ""


class RecordedCall(Frozen):
    """A full prompt as a `FakeLLM` received it."""

    agent: str
    system: str
    user: str
    schema_name: str

    @property
    def text(self) -> str:
        return self.system + "\n" + self.user


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, agent: str, system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
        """Return an instance of `schema` and the usage, or raise an `LLMError`."""
        ...


def prompt_hash(system: str, user: str) -> str:
    """Hash of one full prompt. The same value is logged in decisions and in llm_calls."""
    return sha256_hex(system + "\x1e" + user)


def estimate_tokens(text: str) -> int:
    """Rough upper estimate of the token count (3 characters per token)."""
    return math.ceil(len(text) / 3)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_structured[T: BaseModel](text: str, schema: type[T], usage: Usage | None = None) -> T:
    """Parse model text as JSON and validate it. Raises `InvalidOutput` on any failure."""
    try:
        return schema.model_validate(json.loads(_strip_fence(text)))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise InvalidOutput(f"output is not a valid {schema.__name__}: {exc}", text, usage) from exc


Scripted = Any  # a model instance, a dict, raw text, GARBAGE, an Exception, or a callable
Responder = Callable[[str, str, str, type[BaseModel]], Scripted]


class FakeLLM:
    """A scripted model for tests. No network, deterministic.

    Script it per agent with a queue (`fake.queue("planner", proposal, GARBAGE, error)`) or a
    callable (`fake.on("planner", lambda agent, system, user, schema: ...)`). A scripted item may
    be an instance of the schema, a dict to validate, raw text to parse, `GARBAGE`, or an
    exception to raise. When an agent's queue is empty and it has no callable, `complete` raises
    `LLMError`, which is what a dead provider looks like to the caller.

    `calls` keeps every full prompt. `prompts()` returns them as text for the firewall canary.
    """

    def __init__(self, model: str = FAKE_MODEL, output_tokens: int = 200) -> None:
        self.model = model
        self.output_tokens = output_tokens
        self.calls: list[RecordedCall] = []
        self._queues: dict[str, deque[Scripted]] = {}
        self._responders: dict[str, Responder] = {}

    def queue(self, agent: str, *responses: Scripted) -> FakeLLM:
        self._queues.setdefault(agent, deque()).extend(responses)
        return self

    def on(self, agent: str, responder: Responder) -> FakeLLM:
        self._responders[agent] = responder
        return self

    def prompts(self, agent: str | None = None) -> list[str]:
        return [c.text for c in self.calls if agent is None or c.agent == agent]

    def _next(self, agent: str, system: str, user: str, schema: type[BaseModel]) -> Scripted:
        queue = self._queues.get(agent)
        if queue:
            return queue.popleft()
        if agent in self._responders:
            return self._responders[agent](agent, system, user, schema)
        raise LLMError(f"FakeLLM has no scripted response left for agent {agent!r}")

    def complete(self, agent: str, system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
        self.calls.append(
            RecordedCall(agent=agent, system=system, user=user, schema_name=schema.__name__)
        )
        usage = Usage(
            model=self.model, input_tokens=estimate_tokens(system + user),
            output_tokens=self.output_tokens,
        )
        item = self._next(agent, system, user, schema)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, schema):
            return item, usage
        if isinstance(item, BaseModel):
            item = item.model_dump(mode="json")
        text = item if isinstance(item, str) else json.dumps(item)
        return parse_structured(text, schema, usage), usage


class OpenAIClient:
    """Real client for the OpenAI chat completions API with a JSON schema response format.

    UNVERIFIED against the live API. The request shape follows the public documentation; the
    parsing is tested with an injected fake SDK object only. Run one manual call before relying
    on it. The key is read from the environment and never logged or stored.
    """

    def __init__(
        self,
        models: Mapping[str, str | None],
        *,
        max_output_tokens: int = 4000,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 120.0,
        strict_schema: bool = False,
        client: Any = None,
    ) -> None:
        self._models = dict(models)
        self._max_output_tokens = max_output_tokens
        self._api_key_env = api_key_env
        self._timeout = timeout_seconds
        self._strict = strict_schema
        self._client = client

    def model_for(self, agent: str) -> str:
        model = self._models.get(agent)
        if not model:
            raise ModelNotConfigured(f"no model set for agent {agent!r}")
        return model

    def _sdk(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:
            raise ModelNotConfigured(
                "the `openai` package is not installed. It is not a repo dependency: "
                "install it in the environment that runs the live loop."
            ) from exc
        key = os.environ.get(self._api_key_env)
        if not key:
            raise ModelNotConfigured(f"environment variable {self._api_key_env} is not set")
        self._client = openai.OpenAI(api_key=key, timeout=self._timeout)
        return self._client

    def _request(self, agent: str, system: str, user: str, schema: type[BaseModel]) -> Any:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(),
                "strict": self._strict,
            },
        }
        return self._sdk().chat.completions.create(
            model=self.model_for(agent),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=response_format,
            max_completion_tokens=self._max_output_tokens,
        )

    def complete(self, agent: str, system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
        model = self.model_for(agent)
        try:
            response = self._request(agent, system, user, schema)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        usage = Usage(
            model=model,
            input_tokens=int(getattr(response.usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(response.usage, "completion_tokens", 0) or 0),
        )
        text = response.choices[0].message.content or ""
        return parse_structured(text, schema, usage), usage


def _utc_now() -> datetime:
    return datetime.now(UTC)


_REPAIR_NOTE = (
    "\n\nYour previous answer was rejected because it did not match the output schema:\n"
    "{error}\nAnswer again with one JSON object that matches the schema and nothing else."
)


class BudgetedLLM:
    """The guard every agent calls. Implements `LLMClient`.

    Before each attempt it estimates the worst-case cost (prompt tokens plus `max_output_tokens`
    at the model's price) and raises `BudgetExceeded` WITHOUT calling the model if spending so far
    plus that estimate would pass `cap_usd`. Spending so far is rebuilt from `llm_calls.jsonl`, so
    the cap survives a restart. Invalid output is retried once with the validation error appended.
    Every attempt, successful or not, is one line in `llm_calls.jsonl`.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        cap_usd: float,
        log_path: str | Path,
        prices: Mapping[str, ModelPrice],
        max_output_tokens: int = 4000,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.inner = inner
        self.cap_usd = cap_usd
        self.log_path = Path(log_path)
        self._prices = dict(prices)
        self._max_output_tokens = max_output_tokens
        self._now = now
        self.spent_usd = self._ledger_spent()

    def _ledger_spent(self) -> float:
        """Spending so far, rebuilt from the ledger. Other guards on the same file (another entry
        point) count too, so it is re-read before every call (RT-32)."""
        return sum(float(r.get("cost_usd", 0.0)) for r in read_jsonl(self.log_path).records)

    @property
    def remaining_usd(self) -> float:
        """What is left of the cap on the shared ledger, as of now."""
        self.spent_usd = self._ledger_spent()
        return max(0.0, self.cap_usd - self.spent_usd)

    def _model(self, agent: str) -> str:
        if hasattr(self.inner, "model_for"):
            return self.inner.model_for(agent)
        return str(getattr(self.inner, "model", FAKE_MODEL))

    def _price(self, model: str) -> ModelPrice:
        if model not in self._prices:
            raise ModelNotConfigured(f"no price set for model {model!r}: refusing to call it")
        return self._prices[model]

    def estimate_usd(self, agent: str, system: str, user: str) -> float:
        """Worst-case cost of one attempt with this prompt."""
        price = self._price(self._model(agent))
        return price.cost(estimate_tokens(system + user), self._max_output_tokens)

    def _check_cap(self, agent: str, system: str, user: str) -> None:
        self.spent_usd = self._ledger_spent()
        estimate = self.estimate_usd(agent, system, user)
        if self.spent_usd + estimate > self.cap_usd:
            raise BudgetExceeded(
                f"model budget: spent {self.spent_usd:.4f} + estimate {estimate:.4f} "
                f"> cap {self.cap_usd:.4f} USD; call for {agent} not made"
            )

    def _log(self, agent: str, phash: str, schema: type[BaseModel], usage: Usage | None,
             attempt: int, error: str = "") -> float:
        model = usage.model if usage else self._model(agent)
        tokens = (usage.input_tokens, usage.output_tokens) if usage else (0, 0)
        cost = self._price(model).cost(*tokens)
        self.spent_usd += cost
        record = LLMCallRecord(
            ts=utc_iso(self._now()), agent=agent, prompt_hash=phash, model=model,
            schema_name=schema.__name__, input_tokens=tokens[0], output_tokens=tokens[1],
            cost_usd=cost, ok=not error, attempt=attempt, error=error[:500],
        )
        append_jsonl(self.log_path, record)
        return cost

    def _attempt(self, agent: str, system: str, user: str, schema: type[T],
                 attempt: int) -> tuple[T, Usage]:
        self._check_cap(agent, system, user)
        phash = prompt_hash(system, user)
        try:
            result, usage = self.inner.complete(agent, system, user, schema)
        except InvalidOutput as exc:
            self._log(agent, phash, schema, exc.usage, attempt, error=f"invalid output: {exc}")
            raise
        except LLMError as exc:
            self._log(agent, phash, schema, None, attempt, error=str(exc))
            raise
        except Exception as exc:
            self._log(agent, phash, schema, None, attempt, error=f"{type(exc).__name__}: {exc}")
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        cost = self._log(agent, phash, schema, usage, attempt)
        update = {"cost_usd": cost, "prompt_hash": phash, "attempts": attempt}
        return result, usage.model_copy(update=update)

    def complete(self, agent: str, system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
        try:
            return self._attempt(agent, system, user, schema, attempt=1)
        except InvalidOutput as exc:
            repair = user + _REPAIR_NOTE.format(error=str(exc)[:600])
        return self._attempt(agent, system, repair, schema, attempt=2)


REPO_ROOT = Path(__file__).resolve().parents[3]


def ledger_path(cfg: AgentConfig) -> Path:
    """The one model-spend ledger, `paths.llm_calls` of the config (RT-32). A relative `root`
    (the committed config says ".") resolves against the repository, not the working
    directory, so every entry point finds the same file wherever it is started."""
    if Path(cfg.root).is_absolute():
        return cfg.path("llm_calls")
    path = Path(cfg.paths.llm_calls)
    return path if path.is_absolute() else (REPO_ROOT / cfg.root / path).resolve()


def budgeted_llm(cfg: AgentConfig, agents: tuple[str, ...],
                 inner: LLMClient | None = None) -> BudgetedLLM:
    """The guard an entry point builds: the config's models, prices and cap, on the shared
    ledger. `inner` replaces the OpenAI client (tests pass a `FakeLLM`)."""
    if inner is None:
        inner = OpenAIClient(cfg.require_models(agents), max_output_tokens=cfg.llm_max_output_tokens)
    return BudgetedLLM(inner, cap_usd=cfg.model_budget_usd, log_path=ledger_path(cfg),
                       prices=cfg.prices, max_output_tokens=cfg.llm_max_output_tokens)
