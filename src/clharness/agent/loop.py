"""Entry point for the agent loop.

    python -m clharness.agent.loop --executor sim --out results/agent-sim

`--executor sim` runs the whole loop against the simulator (fake result rows with planted
effects, on a fake clock, so it finishes in seconds and spends nothing). The real executor is
wired in when tasks A1 and A6 land. Running the same command again resumes from state.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import FileMemory, LLMAnalyst, LLMPlanner
from .conductor import Conductor
from .config import load_agent_config
from .contracts import read_jsonl
from .gatekeeper import Gatekeeper
from .llm import BudgetedLLM, LLMClient, OpenAIClient
from .operator import Operator
from .roles import FakeClock, StubAnalyst, StubPlanner
from .sim import SimExecutor, World, default_catalog, default_prereg, default_stream


def build_sim(out: Path, seed: int, compute_cap_usd: float, concurrency: int = 8,
              clock: FakeClock | None = None, fault=None, infra_fail_rate: float = 0.1,
              gatekeeper: Gatekeeper | None = None, llm: LLMClient | None = None,
              memory: bool = True) -> Conductor:
    clock = clock or FakeClock()
    world = World(seed, infra_fail_rate=infra_fail_rate)
    rows_path = out / "rows.jsonl"
    executor = SimExecutor(world, rows_path, clock, concurrency=concurrency)
    return Conductor(
        out, rows_path, executor=executor, clock=clock,
        planner=LLMPlanner(llm) if llm else StubPlanner(),
        analyst=LLMAnalyst(llm, clock=clock) if llm else StubAnalyst(clock),
        memory=FileMemory(out / "memory") if memory else None,
        gatekeeper=gatekeeper or Gatekeeper(), operator=Operator(executor, clock, rows_path),
        prereg=default_prereg(world), catalog=default_catalog(), stream=default_stream(),
        compute_cap_usd=compute_cap_usd, concurrency=concurrency, fault=fault,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--executor", choices=["sim"], required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for rows, state, log")
    parser.add_argument("--seed", type=int, default=0, help="simulator world seed")
    parser.add_argument("--compute-cap-usd", type=float, default=700.0)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--llm", choices=["none", "openai"], default="none",
                        help="model for the Planner and Analyst; none = deterministic roles")
    parser.add_argument("--config", type=Path, default=None,
                        help="agent config YAML (models, prices, model budget) for --llm openai")
    args = parser.parse_args(argv)

    llm = None
    if args.llm == "openai":
        if args.config is None:
            parser.error("--llm openai needs --config")
        cfg = load_agent_config(args.config)
        models = cfg.require_models(("planner", "analyst"))
        llm = BudgetedLLM(
            OpenAIClient(models, max_output_tokens=cfg.llm_max_output_tokens),
            cap_usd=cfg.model_budget_usd, log_path=args.out / "llm_calls.jsonl",
            prices=cfg.prices, max_output_tokens=cfg.llm_max_output_tokens,
        )
    conductor = build_sim(args.out, args.seed, args.compute_cap_usd, llm=llm)
    state = conductor.run(args.max_transitions)
    decisions = read_jsonl(conductor.log_path).records
    for d in decisions:
        data = d.get("data", {})
        accepted = {True: " accepted", False: " REJECTED", None: ""}[data.get("accepted")]
        choice = json.dumps(data.get("choice"))[:110]
        print(f"{d['ts'][11:19]}  {d['state']:<24}{accepted}  {choice}")
    print(f"\nstate: {state.state.id}   decisions: {len(decisions)}   "
          f"compute spent: ${state.compute_usd_spent:.2f}")
    if state.halted_reason:
        print(f"halted: {state.halted_reason}")
    print(f"log: {conductor.log_path}\nrows: {conductor.rows_path}")
    return 0 if state.state.id in ("done",) or args.max_transitions else 1


if __name__ == "__main__":
    raise SystemExit(main())
