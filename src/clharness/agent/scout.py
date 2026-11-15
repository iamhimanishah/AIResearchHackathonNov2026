"""Run the Scout step once, live: Exa search, then a model picks at most one candidate.

    python -m clharness.agent.scout --search-only                 # Exa only, prints sources
    python -m clharness.agent.scout --config agent.yaml           # search + model, writes a card

Needs EXA_API_KEY (and OPENAI_API_KEY unless --search-only) in `.env`. The candidate card and
every source it saw are written to --out as one JSON file; a human reviews the card before
anyone builds the technique.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import env  # noqa: F401 - loads .env
from .agents import LLMScout, write_scout_result
from .config import load_agent_config
from .llm import BudgetedLLM, OpenAIClient
from .roles import ScoutRequest
from .search import ExaSearch
from .sim import default_catalog


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
    models = cfg.require_models(("scout",))
    llm = BudgetedLLM(
        OpenAIClient(models, max_output_tokens=cfg.llm_max_output_tokens),
        cap_usd=cfg.model_budget_usd, log_path=args.out / "llm_calls.jsonl",
        prices=cfg.prices, max_output_tokens=cfg.llm_max_output_tokens,
    )
    result = LLMScout(llm, search).scout(ScoutRequest(catalog=default_catalog()))
    path = write_scout_result(result, args.out)
    if result.candidate:
        c = result.candidate
        print(f"candidate: {c.name}\n  {c.description}\n  sources:")
        for s in result.sources:
            if s.used:
                print(f"    {s.url}")
    else:
        print(f"skipped: {result.skipped_reason}")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
