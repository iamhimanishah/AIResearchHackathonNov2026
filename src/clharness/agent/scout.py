"""Run the Scout step once, live: Exa search, then a model drafts a ranked shortlist.

    python -m clharness.agent.scout --search-only                 # Exa only, prints sources
    python -m clharness.agent.scout --config configs/agent.yaml   # search + model, writes a file

Needs EXA_API_KEY (and OPENAI_API_KEY unless --search-only) in `.env`. Claim cards and candidates
are checked in code against the real plugin catalog, exactly as in the loop's real mode. The
shortlist and every source it saw are written to --out as one JSON file; a person reviews a
candidate before anyone builds the technique. Nothing is launched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import env  # noqa: F401 - loads .env
from .agents import LLMScout, write_scout_result
from .catalog_real import bench_problems, design_claim_arms, real_catalog
from .catalog_real import stream as bench_stream
from .config import load_agent_config
from .llm import budgeted_llm
from .roles import ScoutRequest, ScoutResult
from .search import ExaSearch


def real_request() -> ScoutRequest:
    """The request the loop's real mode sends: real catalog, frozen stream, PREREG claim arms."""
    return ScoutRequest(catalog=real_catalog(), stream=bench_stream(), seeds=(1, 2),
                        design_arms=design_claim_arms())


def summary(result: ScoutResult) -> str:
    """What the CLI prints for a result: both lists, what failed a check, the sources used."""
    lines = []
    for card in result.claims:
        repeat = f" (repeats PREREG claim {card.duplicate_of})" if card.duplicate_of else ""
        lines.append(f"claim {card.id}{repeat}: {card.claim}\n  test {card.test_arm} "
                     f"vs control {card.control_arm or 'no mitigation'}")
    for cand in result.candidates:
        lines.append(f"candidate: {cand.name}\n  {cand.description}")
    lines += [f"rejected: {why}" for why in result.rejected]
    if not result.claims and not result.candidates:
        lines.append(f"skipped: {result.skipped_reason}")
    used = [s.url for s in result.sources if s.used]
    if used:
        lines.append("sources used:\n" + "\n".join(f"  {u}" for u in used))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/agent/candidates"))
    parser.add_argument("--config", type=Path, default=None,
                        help="agent config YAML with a model and price for the scout")
    parser.add_argument("--search-only", action="store_true",
                        help="run the Exa queries and print the sources; no model call")
    args = parser.parse_args(argv)

    search = ExaSearch()
    if args.search_only:
        sources, errors = LLMScout(llm=None, search=search).gather()  # type: ignore[arg-type]
        for err in errors:
            print(f"search error: {err}")
        for s in sources:
            print(f"{s.id}  {s.published or '-':<10.10}  {s.title[:70]}\n    {s.url}")
        print(f"\n{len(sources)} sources")
        return 0 if sources else 1

    if args.config is None:
        parser.error("a model call needs --config (or use --search-only)")
    cfg = load_agent_config(args.config)
    llm = budgeted_llm(cfg, ("scout",))  # the loop's model ledger, shared (RT-32)
    print(f"model ledger {llm.log_path}: spent ${llm.cap_usd - llm.remaining_usd:.2f} "
          f"of ${llm.cap_usd:.2f}")
    result = LLMScout(llm, search, runnable=bench_problems).scout(real_request())
    path = write_scout_result(result, args.out)
    print(summary(result))
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
