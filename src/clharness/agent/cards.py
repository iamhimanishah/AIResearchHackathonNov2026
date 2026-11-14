"""Agent cards and skills: plain markdown files, loaded and hashed.

A card is `cards/<agent>.md`: YAML frontmatter (the agent's context-map row, tools and budget)
and a body that is the system prompt. A skill is `skills/<name>.md`: one short procedure, used
for one state. The content hash of each is what the decision log records, so a judge can see
which version of a prompt made which choice.

Card frontmatter keys (all required except `parent_version`):

    agent: planner            # one of planner, analyst, scout, reporter
    reads: [phase, catalog]   # context blocks, names from context.CONTEXT_MAP
    never_reads: [rows_report]
    writes: proposal
    tools: []
    budget: {max_calls_per_state: 3}
    version: 1
    parent_version: null      # the version this one was evolved from (set by the optimizer)

`SKILLS` maps `(agent, state)` to skill names. A lookup tries the exact state id first
(`"screen/plan"`), then the bare step (`"plan"`).

Note for other parts: `cards.py` (this module) and the data directory `cards/` share a name.
That works because `cards/` holds only markdown. Never add an `__init__.py` to `cards/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError, field_validator

from .config import PACKAGE_DIR
from .contracts import MODEL_AGENTS, Frozen, sha256_hex

DEFAULT_CARDS_DIR = PACKAGE_DIR / "cards"
DEFAULT_SKILLS_DIR = PACKAGE_DIR / "skills"

# (agent, state id or bare step) -> skill names, in the order they enter the prompt.
SKILLS: dict[tuple[str, str], tuple[str, ...]] = {
    ("planner", "screen/plan"): ("plan_screen",),
    ("planner", "replicate/plan"): ("plan_replicate",),
    ("planner", "combine/plan"): ("plan_combine",),
    ("analyst", "read"): ("read_findings",),
    ("scout", "scout"): ("scout_candidate",),
    ("reporter", "report"): ("write_report",),
}


class CardError(ValueError):
    """A card or skill file is missing, malformed or inconsistent."""


def content_hash(text: str) -> str:
    """sha256 of the text with line endings normalised, so a checkout cannot change the hash."""
    return sha256_hex(text.replace("\r\n", "\n").replace("\r", "\n"))


class Card(Frozen):
    agent: str
    reads: tuple[str, ...]
    never_reads: tuple[str, ...]
    writes: str
    tools: tuple[str, ...]
    budget: dict[str, float]
    version: int = Field(ge=1)
    parent_version: int | None = None
    body: str  # the system prompt
    hash: str  # content hash of the whole file: frontmatter and body

    @field_validator("agent")
    @classmethod
    def _known_agent(cls, value: str) -> str:
        if value not in MODEL_AGENTS:
            raise ValueError(f"agent must be one of {MODEL_AGENTS}")
        return value

    @field_validator("body")
    @classmethod
    def _body_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the card body (system prompt) is empty")
        return value


class Skill(Frozen):
    name: str
    body: str
    hash: str


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split `---` YAML frontmatter from the markdown body."""
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise CardError("missing YAML frontmatter (the file must start with ---)")
    head, sep, body = text[4:].partition("\n---\n")
    if not sep:
        raise CardError("frontmatter is not closed with ---")
    try:
        meta = yaml.safe_load(head) or {}
    except yaml.YAMLError as exc:
        raise CardError(f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise CardError("frontmatter must be a mapping")
    return meta, body.strip() + "\n"


def parse_card(text: str) -> Card:
    """Parse card text. Unknown frontmatter keys are errors, so a typo cannot hide a rule."""
    meta, body = split_frontmatter(text)
    try:
        return Card(**meta, body=body, hash=content_hash(text))
    except (ValidationError, TypeError) as exc:
        raise CardError(f"card frontmatter is invalid: {exc}") from exc


def render_card(card: Card) -> str:
    """Card back to file text (used by the optimizer to write a new version)."""
    meta = card.model_dump(mode="json", exclude={"body", "hash"})
    return "---\n" + yaml.safe_dump(meta, sort_keys=False).strip() + "\n---\n" + card.body


def load_card(path: str | Path) -> Card:
    path = Path(path)
    if not path.exists():
        raise CardError(f"card not found: {path.name}")
    return parse_card(path.read_text(encoding="utf-8"))


def load_skill(path: str | Path) -> Skill:
    path = Path(path)
    if not path.exists():
        raise CardError(f"skill not found: {path.name}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise CardError(f"skill is empty: {path.name}")
    return Skill(name=path.stem, body=text.strip() + "\n", hash=content_hash(text))


def skill_names(agent: str, state_id: str) -> tuple[str, ...]:
    """Skills for an agent in a state: exact state id first, then the bare step, else none."""
    step = state_id.rpartition("/")[2]
    return SKILLS.get((agent, state_id), SKILLS.get((agent, step), ()))


class CardStore:
    """Loads cards and skills from two directories. Files are re-read on every call."""

    def __init__(
        self, cards_dir: str | Path | None = None, skills_dir: str | Path | None = None
    ) -> None:
        self.cards_dir = Path(cards_dir) if cards_dir else DEFAULT_CARDS_DIR
        self.skills_dir = Path(skills_dir) if skills_dir else DEFAULT_SKILLS_DIR

    def card(self, agent: str) -> Card:
        card = load_card(self.cards_dir / f"{agent}.md")
        if card.agent != agent:
            raise CardError(f"{agent}.md declares agent {card.agent!r}")
        return card

    def skill(self, name: str) -> Skill:
        return load_skill(self.skills_dir / f"{name}.md")

    def skills_for(self, agent: str, state_id: str) -> tuple[Skill, ...]:
        return tuple(self.skill(name) for name in skill_names(agent, state_id))

    def hashes(self) -> dict[str, str]:
        """`card:<agent>` and `skill:<name>` -> content hash, for every file the loop can use."""
        out = {f"card:{agent}": self.card(agent).hash for agent in MODEL_AGENTS}
        for names in SKILLS.values():
            out.update({f"skill:{name}": self.skill(name).hash for name in names})
        return out

    def problems(self) -> list[str]:
        """What PREFLIGHT must refuse: a missing or malformed card or skill. Empty means ready."""
        found = []
        for agent in MODEL_AGENTS:
            try:
                self.card(agent)
            except CardError as exc:
                found.append(f"card {agent}: {exc}")
        for name in sorted({n for names in SKILLS.values() for n in names}):
            try:
                self.skill(name)
            except CardError as exc:
                found.append(f"skill {name}: {exc}")
        return found
