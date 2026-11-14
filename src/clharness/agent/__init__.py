"""The agent package: the research loop that plans, launches, reads and decides (workstream 5).

Start with `docs/agent/ARCHITECTURE.md` (the design) and `docs/agent/kernel.md` (the shared
contracts in this package). The kernel modules are:

    contracts  states  config  llm  cards  context  roles  sim

Importing this package pulls in the standard library, pydantic and pyyaml only. It imports no
training code and no network client.
"""

from .contracts import (
    Catalog,
    Decision,
    DecisionLog,
    Finding,
    FirewallError,
    FreezeToken,
    Phase,
    Prereg,
    Proposal,
    ReportRow,
    ResultRow,
    RunSpec,
    SelectionRow,
    Verdict,
    mint_freeze_token,
    spec_hash,
    to_report_view,
    to_selection_view,
)
from .states import State, Step

__all__ = [
    "Catalog", "Decision", "DecisionLog", "Finding", "FirewallError", "FreezeToken", "Phase",
    "Prereg", "Proposal", "ReportRow", "ResultRow", "RunSpec", "SelectionRow", "State", "Step",
    "Verdict", "mint_freeze_token", "spec_hash", "to_report_view", "to_selection_view",
]
