"""Entry point for the agent loop.

    python -m clharness.agent.loop --executor sim --out results/agent-sim
    python -m clharness.agent.loop --executor real --prereg thresholds.yaml \
        --rows results/agent/rows.jsonl --out results/agent/loop

`--executor sim` runs the whole loop against the simulator (fake result rows with planted
effects, on a fake clock, so it finishes in seconds and spends nothing). `--scout exa` adds the
Scout after replicate: one live Exa search, a ranked shortlist of claim cards and candidates (for
human review), every source in the decision log (needs `--llm openai`).

`--executor real` runs the loop on real rows and launches nothing. It reads the rows file the
pre-registered grid filled (`launch_design`), plans and gates against the real plugin catalog
(`catalog_real.real_catalog`, with the bench check in the Gatekeeper), runs the Scout on that
catalog, and writes every accepted spec to `<out>/proposed_specs.jsonl` as a proposal. Its
executor refuses `submit`, so it needs no Modal deploy and cannot spend GPU money: every launch
goes through `launch_design`. Claim cards go to `--explore-inbox` (default
`results/agent/explore/literature.jsonl`, the file `explore.read_inbox` reads) for the explore
step; this loop tests none of them. It stops after FREEZE (selection half only): `--open-report`
is needed to open the report half of real rows. Running the same command again resumes from
state.json. It refuses a rows file of exploratory runs (`rows.explore.jsonl`).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .agents import FileMemory, LLMAnalyst, LLMPlanner, LLMScout
from .catalog_real import (
    bench_problems,
    design_claim_arms,
    design_exploratory_configs,
    real_catalog,
)
from .catalog_real import stream as bench_stream
from .conductor import Conductor
from .config import load_agent_config
from .contracts import Prereg, PreregError, load_prereg, read_jsonl
from .explore import INBOX_NAME
from .gatekeeper import Gatekeeper
from .llm import BudgetedLLM, LLMClient, OpenAIClient
from .operator import Operator
from .roles import Clock, FakeClock, Scout, StubAnalyst, StubPlanner, SystemClock
from .search import ExaSearch
from .sim import SimExecutor, World, default_catalog, default_prereg, default_stream
from .states import REPORT

SIM_CAP_USD = 700.0  # the simulator's default cap: fake money, DIRECTION's team-wide number


def build_sim(out: Path, seed: int, compute_cap_usd: float, concurrency: int = 8,
              clock: FakeClock | None = None, fault=None, infra_fail_rate: float = 0.1,
              gatekeeper: Gatekeeper | None = None, llm: LLMClient | None = None,
              memory: bool = True, scout: Scout | None = None,
              explore_inbox: Path | None = None) -> Conductor:
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
        compute_cap_usd=compute_cap_usd, concurrency=concurrency, fault=fault, scout=scout,
        explore_inbox=explore_inbox,
    )


def build_real(out: Path, prereg: Prereg, rows_path: Path, *, compute_cap_usd: float,
               gpu: str = "H100", clock: Clock | None = None, llm: LLMClient | None = None,
               scout: Scout | None = None, memory: bool = True,
               deadline: datetime | None = None, fault=None,
               explore_inbox: Path | None = None, reserve_fraction: float = 0.3) -> Conductor:
    """The loop on real rows, read-only: a `ReadOnlyExecutor` (refuses every launch), the real
    catalog, the bench check in the gate, the frozen stream. Accepted specs become proposals;
    claim cards go to the explore inbox (`explore_inbox`, default `<out>/explore/literature.jsonl`;
    the file `explore.read_inbox` reads) and this loop tests none of them."""
    from .launch_design import MINUTES_PER_RUN
    from .readonly import ReadOnlyExecutor

    rows_path = Path(rows_path)
    if "explore" in rows_path.name:
        raise ValueError(f"{rows_path.name} holds exploratory runs: the loop reads the "
                         "pre-registered rows only")
    clock = clock or SystemClock()
    catalog = real_catalog()
    executor = ReadOnlyExecutor(rows_path, minutes_per_run=MINUTES_PER_RUN, gpu=gpu,
                                catalog=catalog)
    return Conductor(
        out, rows_path, executor=executor, clock=clock,
        planner=LLMPlanner(llm) if llm else StubPlanner(),
        analyst=LLMAnalyst(llm, clock=clock) if llm else StubAnalyst(clock),
        memory=FileMemory(out / "memory") if memory else None,
        gatekeeper=Gatekeeper(runnable=bench_problems),
        operator=Operator(executor, clock, rows_path),
        prereg=prereg, catalog=catalog, stream=bench_stream(),
        compute_cap_usd=compute_cap_usd, reserve_fraction=reserve_fraction, deadline=deadline,
        exploratory_configs=design_exploratory_configs(),
        minutes_per_run=MINUTES_PER_RUN, fault=fault, scout=scout,
        design_arms=design_claim_arms(),
        explore_inbox=explore_inbox or out / "explore" / INBOX_NAME, launches=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--executor", choices=["sim", "real"], required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for rows, state, log")
    parser.add_argument("--rows", type=Path, default=None,
                        help="real: the rows file to read, never written "
                             "(default results/agent/rows.jsonl)")
    parser.add_argument("--prereg", type=Path, default=None,
                        help="real: PREREG thresholds (a .yaml, or a .md with an "
                             "`agent_thresholds` block)")
    parser.add_argument("--open-report", action="store_true",
                        help="real: also run REPORT, which opens the report half of the frozen "
                             "set. Off by default: the loop stops after FREEZE")
    parser.add_argument("--gpu", default="H100", help="real: GPU used to price proposals")
    parser.add_argument("--explore-inbox", type=Path,
                        default=Path("results/agent/explore") / INBOX_NAME,
                        help="real: where the Scout's claim cards are written for the explore "
                             "step (the file explore.read_inbox reads)")
    parser.add_argument("--seed", type=int, default=0, help="simulator world seed")
    parser.add_argument("--compute-cap-usd", type=float, default=None,
                        help="compute cap in dollars. Default: compute_budget_usd from --config; "
                             f"sim without a config: {SIM_CAP_USD:.0f}. real: one of the two "
                             "is required")
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--llm", choices=["none", "openai"], default="none",
                        help="model for the Planner and Analyst; none = deterministic roles")
    parser.add_argument("--scout", choices=["none", "exa"], default="none",
                        help="exa = after replicate, search the web with Exa for one technique "
                             "not in the catalog (needs --llm openai and EXA_API_KEY)")
    parser.add_argument("--config", type=Path, default=None,
                        help="agent config YAML (models, prices, model budget) for --llm openai")
    args = parser.parse_args(argv)

    if args.scout == "exa" and args.llm != "openai":
        parser.error("--scout exa needs --llm openai (the Scout is a model step)")
    llm = None
    if args.llm == "openai" and args.config is None:
        parser.error("--llm openai needs --config")
    cfg = load_agent_config(args.config) if args.config else None
    # the cap: the flag when given, else the config's; never a silent default on real rows
    if args.compute_cap_usd is None:
        if cfg is not None:
            args.compute_cap_usd = cfg.compute_budget_usd
        elif args.executor == "sim":
            args.compute_cap_usd = SIM_CAP_USD
        else:
            parser.error("--executor real needs --compute-cap-usd or a --config with "
                         "compute_budget_usd")
    args.reserve_fraction = cfg.compute_reserve_fraction if cfg is not None else 0.3
    if args.llm == "openai":
        agents = ("planner", "analyst") + (("scout",) if args.scout == "exa" else ())
        models = cfg.require_models(agents)
        llm = BudgetedLLM(
            OpenAIClient(models, max_output_tokens=cfg.llm_max_output_tokens),
            cap_usd=cfg.model_budget_usd, log_path=args.out / "llm_calls.jsonl",
            prices=cfg.prices, max_output_tokens=cfg.llm_max_output_tokens,
        )
    if args.executor == "real":
        return _main_real(args, llm)
    scout = LLMScout(llm, ExaSearch()) if args.scout == "exa" and llm else None
    conductor = build_sim(args.out, args.seed, args.compute_cap_usd, llm=llm, scout=scout)
    return _run_and_print(conductor, args)


def _main_real(args: argparse.Namespace, llm: LLMClient | None) -> int:
    if args.prereg is None:
        print("--executor real needs --prereg (analysis/PREREG.md has no machine-readable "
              "`agent_thresholds` block yet)")
        return 2
    try:
        prereg = load_prereg(args.prereg)
    except PreregError as exc:
        print(f"PREREG: {exc}")
        return 2
    rows = args.rows or Path("results/agent/rows.jsonl")
    scout = (LLMScout(llm, ExaSearch(), runnable=bench_problems)
             if args.scout == "exa" and llm else None)
    try:
        conductor = build_real(args.out, prereg, rows, compute_cap_usd=args.compute_cap_usd,
                               gpu=args.gpu, llm=llm, scout=scout,
                               explore_inbox=args.explore_inbox,
                               reserve_fraction=args.reserve_fraction)
    except ValueError as exc:
        print(f"refused: {exc}")
        return 2
    print(f"read-only: nothing will be launched. cap ${args.compute_cap_usd:.0f} "
          f"(reserve {args.reserve_fraction:.0%}); rows {rows}; catalog "
          f"{list(conductor.catalog.names())}; scout {'on' if scout else 'off'}; proposals go to "
          f"{conductor.proposals_path}; claim cards go to {conductor.explore_inbox}")
    return _run_and_print(conductor, args, stop_before=None if args.open_report else REPORT)


def _run_and_print(conductor: Conductor, args: argparse.Namespace, stop_before=None) -> int:
    state = conductor.run(args.max_transitions, stop_before=stop_before)
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
    if stop_before is not None and state.state == stop_before:
        print(f"stopped before {stop_before.id}: the report half is not opened without "
              "--open-report")
        return 0
    return 0 if state.state.id in ("done",) or args.max_transitions else 1


if __name__ == "__main__":
    raise SystemExit(main())
