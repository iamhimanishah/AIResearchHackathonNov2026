"""Forgetting-reduction techniques, as hook plugins.

A technique is a `BaseHook` subclass with three things:

  name     the string a run spec uses to switch it on (`techniques: [{name: replay, ...}]`)
  Params   a pydantic model of its parameters; unknown keys are errors
  __init__(self, params)   takes the validated Params instance

A technique may also define `config_overrides(params) -> dict`, a nested dict merged into the
run config before training (see `apply_config_overrides`). That is how a technique that is
really a setting, such as per-domain LR re-warm, is still switched on and off by name.

Register it with `@register`, in its own module in this package. Every module here is imported
the first time a technique is looked up, so adding a file is enough. The training loop never
names a technique: `build_hooks` turns a run's `techniques` list into hook instances, and the
caller (`clh train`, the executor) appends them after the measurement hooks.

    @register
    class Replay(BaseHook):
        name = "replay"

        class Params(TechniqueParams):
            share: float = 0.05

        def __init__(self, params):
            self.p = params
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from clharness.train.hooks import BaseHook

if TYPE_CHECKING:
    from clharness.config import RunConfig


class TechniqueParams(BaseModel):
    """Base for every technique's Params: a typo in a parameter name is an error, not a no-op."""

    model_config = ConfigDict(extra="forbid", frozen=True)


REGISTRY: dict[str, type[BaseHook]] = {}
_loaded = False


def _load_plugins() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        for mod in pkgutil.iter_modules(__path__):
            importlib.import_module(f"{__name__}.{mod.name}")


def available() -> dict[str, dict[str, Any]]:
    """Technique name -> JSON Schema of its params (for docs and for the planning agent)."""
    _load_plugins()
    return {name: cls.Params.model_json_schema() for name, cls in sorted(REGISTRY.items())}


def register(cls: type[BaseHook]) -> type[BaseHook]:
    """Class decorator: make `cls` available to run specs under `cls.name`."""
    if not isinstance(getattr(cls, "Params", None), type) or not issubclass(cls.Params, TechniqueParams):
        raise TypeError(f"technique {cls.__name__} needs a Params class derived from TechniqueParams")
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"technique name {cls.name!r} is already registered")
    REGISTRY[cls.name] = cls
    return cls


def _validated(techniques: Iterable[Any]) -> list[tuple[type[BaseHook], TechniqueParams]]:
    """(class, validated params) per technique, sorted by name: the order techniques are listed
    in carries no meaning, so two orderings of one set can never behave differently."""
    _load_plugins()
    out: list[tuple[type[BaseHook], TechniqueParams]] = []
    seen: set[str] = set()
    for t in sorted(techniques, key=lambda t: t.name):
        if t.name in seen:
            raise ValueError(f"technique {t.name!r} is listed twice")
        seen.add(t.name)
        if t.name not in REGISTRY:
            raise ValueError(f"unknown technique {t.name!r}; known: {sorted(REGISTRY)}")
        cls = REGISTRY[t.name]
        try:
            out.append((cls, cls.Params.model_validate(dict(t.params))))
        except ValidationError as e:
            raise ValueError(f"technique {t.name!r}: {e}") from e
    return out


def build_hooks(techniques: Iterable[Any]) -> list[BaseHook]:
    """Turn `[{name, params}, ...]` (RunConfig.techniques / RunSpec.techniques) into hooks.
    Raises ValueError for an unknown name, a repeated name or bad params."""
    return [cls(params) for cls, params in _validated(techniques)]


def apply_config_overrides(cfg: RunConfig) -> RunConfig:
    """The config with every listed technique's `config_overrides` merged in. Call it on the
    final config, after any other overrides: a technique's settings win. Two techniques that
    set the same field to different values are an error."""
    from clharness.config import RunConfig, _deep_merge

    merged: dict[str, Any] = {}
    owner: dict[str, str] = {}
    for cls, params in _validated(cfg.techniques):
        overrides = cls.config_overrides(params) if hasattr(cls, "config_overrides") else {}
        for path, value in _leaves(overrides):
            if path in owner and _get(merged, path) != value:
                raise ValueError(f"techniques {owner[path]!r} and {cls.name!r} both set {'.'.join(path)}")
            owner[path] = cls.name
        merged = _deep_merge(merged, overrides)
    return RunConfig.model_validate(_deep_merge(cfg.dump(), merged)) if merged else cfg


def _leaves(d: dict[str, Any], prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, (*prefix, k))
        else:
            yield (*prefix, k), v


def _get(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    for k in path:
        d = d[k]
    return d

