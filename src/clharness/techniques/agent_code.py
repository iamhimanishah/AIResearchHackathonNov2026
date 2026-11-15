"""A technique written by the research agent, run behind checks.

The explore loop may propose a technique that is not in the catalog by writing one hook class.
This plugin carries that source code in the run spec (`params.source`), so the spec hash covers
the code, every result row records exactly what ran, and no redeploy is needed per technique.

The code is model-written and is treated as untrusted. Three checks stand between a proposal
and a full run, each cheaper than the next thing it protects:

  1. `validate_source` (here; no execution). An allow-list on the syntax tree: only `torch`,
     `math` and the hook base class may be imported; no file, process, network or reflection
     access; no names or attributes beginning with two underscores; no `while`; one class, which
     subclasses `BaseHook`. It runs at the gate, again in the tiny-model check, and again inside
     the GPU container before the code is executed.
  2. `clharness.techniques.agent_check` (the laptop, seconds, in a subprocess with a timeout).
     The class trains a tiny random model for two domains: it must not raise, the loss must stay
     finite, and the run must spend exactly the gradient tokens the plain run spends.
  3. A probe run on a real GPU: the frozen bench cut to four optimizer steps per domain (about
     35 cents). Only a finished probe with finite numbers lets the two full runs start.

What the code may do is what any hook may do (clharness/train/hooks.py): change or freeze the
model before the optimizer is built (`configure_model`), add a term to the loss (`extra_loss`),
replace or drop a micro-batch (`select_batch`), act at domain boundaries, read `ctx.model`.

Limits, stated plainly: the allow-list is a seatbelt for a weekend project, not a sandbox
proven against a determined attacker. The code runs in a cloud container that holds a Hugging
Face read token and the run volumes. Everything from this path is exploratory.
"""

from __future__ import annotations

import ast
import math
from typing import Any

import torch
from pydantic import Field

from clharness.techniques import TechniqueParams, register
from clharness.train.hooks import BaseHook, CheckpointInfo, RunContext, StepStats

MAX_SOURCE_CHARS = 6000
ALLOWED_IMPORTS = {"torch", "torch.nn", "torch.nn.functional", "math", "clharness.train.hooks"}
ALLOWED_FROM = {
    "torch": {"nn", "Tensor"}, "torch.nn": {"functional"}, "clharness.train.hooks": {"BaseHook"},
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "input", "breakpoint", "exit", "quit", "type", "memoryview", "help", "dir", "object", "classmethod",
}
FORBIDDEN_ATTRS = {
    "save", "load", "hub", "jit", "distributed", "multiprocessing", "system", "popen", "environ",
    "remove", "unlink", "rmtree", "urlopen", "request", "socket", "subprocess", "from_file", "tofile",
    "output_dir", "data_manifest", "eval_manifest", "tokenizer", "save_pretrained", "push_to_hub",
    "cuda", "backends", "utils", "serialization", "storage",
}
FORBIDDEN_NODES = (ast.While, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
                   ast.Await, ast.Yield, ast.YieldFrom, ast.Lambda, ast.Try, ast.Raise, ast.Delete,
                   ast.Starred, ast.NamedExpr)
HOOK_METHODS = (
    "configure_model", "extra_loss", "on_run_start", "on_domain_start", "select_batch", "on_step_end",
    "wants_activations", "on_micro_batch", "on_checkpoint", "on_domain_end", "on_run_end",
)
_SAFE_BUILTINS = {
    n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
    for n in ("len", "range", "min", "max", "sum", "abs", "float", "int", "bool", "str", "list", "dict", "tuple",
              "set", "enumerate", "zip", "sorted", "isinstance", "any", "all", "round", "super", "reversed",
              "hasattr", "print", "True", "False", "None", "__build_class__", "ValueError", "RuntimeError")
    if (n in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, n))
}


def validate_source(source: str) -> list[str]:
    """Problems with a proposed technique's source. Empty means it may be executed."""
    if len(source) > MAX_SOURCE_CHARS:
        return [f"source is {len(source)} characters; the limit is {MAX_SOURCE_CHARS}"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"does not parse: {exc.msg} (line {exc.lineno})"]
    problems: list[str] = []
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    for node in tree.body:
        constant = isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)) and not constant and not docstring:
            problems.append(f"line {node.lineno}: only imports, constants and one class are allowed at the top level")
    if len(classes) != 1:
        problems.append(f"expected exactly one class, found {len(classes)}")
    elif [b.id for b in classes[0].bases if isinstance(b, ast.Name)] != ["BaseHook"] or classes[0].keywords \
            or classes[0].decorator_list:
        problems.append("the class must subclass BaseHook, and nothing else, with no decorators")
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, FORBIDDEN_NODES):
            problems.append(f"line {line}: {type(node).__name__} is not allowed")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    problems.append(f"line {line}: import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            allowed = ALLOWED_FROM.get(node.module or "", set())
            for alias in node.names:
                if node.level or alias.name not in allowed:
                    problems.append(f"line {line}: 'from {node.module} import {alias.name}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                problems.append(f"line {line}: the name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                problems.append(f"line {line}: the attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("__") and node.name != "__init__":
                problems.append(f"line {line}: the method {node.name!r} is not allowed")
            if node.decorator_list:
                problems.append(f"line {line}: decorators are not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            problems.append(f"line {line}: a string containing '__' is not allowed")
    return problems[:12]


def _import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    if level or name not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in agent-written code")
    return __import__(name, globals_, locals_, fromlist, level)


def load_hook_class(source: str) -> type[BaseHook]:
    """Validate, then execute the source with a minimal set of builtins, and return its class."""
    problems = validate_source(source)
    if problems:
        raise ValueError("agent-written technique refused: " + "; ".join(problems))
    namespace: dict[str, Any] = {"__builtins__": {**_SAFE_BUILTINS, "__import__": _import},
                                 "__name__": "agent_technique"}
    exec(compile(source, "<agent technique>", "exec"), namespace)  # noqa: S102 - checked above
    found = [v for v in namespace.values()
             if isinstance(v, type) and issubclass(v, BaseHook) and v is not BaseHook]
    if len(found) != 1:
        raise ValueError("agent-written technique refused: it did not define exactly one BaseHook subclass")
    return found[0]


@register
class AgentCode(BaseHook):
    name = "agent_code"
    hidden = True  # runnable from a spec, but not listed by techniques.available()

    class Params(TechniqueParams):
        source: str = Field(min_length=1)  # one BaseHook subclass; see the module docstring
        params: dict[str, Any] = Field(default_factory=dict)  # keyword arguments for its __init__
        title: str = ""  # a short human-readable name for tables

    def __init__(self, params: Params):
        self.p = params
        cls = load_hook_class(params.source)
        try:
            self.inner = cls(**params.params)
        except Exception as exc:
            raise ValueError(f"agent-written technique refused: its constructor raised "
                             f"{type(exc).__name__}: {exc}") from exc

    # Every hook point is forwarded. A value that is not finite where a number is expected would
    # otherwise only show up as a NaN loss many steps later, so the loss term is checked here.

    def configure_model(self, ctx: RunContext):
        return self.inner.configure_model(ctx)

    def extra_loss(self, ctx: RunContext, batch: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
        term = self.inner.extra_loss(ctx, batch, logits, labels)
        if term is not None and not math.isfinite(float(term.detach())):
            raise RuntimeError("agent-written technique returned a loss term that is not finite")
        return term

    def on_run_start(self, ctx: RunContext):
        return self.inner.on_run_start(ctx)

    def on_domain_start(self, ctx: RunContext, domain: str):
        return self.inner.on_domain_start(ctx, domain)

    def select_batch(self, ctx: RunContext, batch: torch.Tensor):
        return self.inner.select_batch(ctx, batch)

    def on_step_end(self, ctx: RunContext, stats: StepStats):
        return self.inner.on_step_end(ctx, stats)

    def wants_activations(self) -> list[int]:
        return self.inner.wants_activations()

    def on_micro_batch(self, ctx: RunContext, step: int, domain_index: int, chunk_ids, batch, logits, seq_loss, acts):
        return self.inner.on_micro_batch(ctx, step, domain_index, chunk_ids, batch, logits, seq_loss, acts)

    def on_checkpoint(self, ctx: RunContext, ckpt: CheckpointInfo):
        return self.inner.on_checkpoint(ctx, ckpt)

    def on_domain_end(self, ctx: RunContext, domain: str):
        return self.inner.on_domain_end(ctx, domain)

    def on_run_end(self, ctx: RunContext):
        return self.inner.on_run_end(ctx)
