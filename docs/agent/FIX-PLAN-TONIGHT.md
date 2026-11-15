# Agent workstream: what we are fixing tonight (for review)

Owner: brendan. Written 14 November, evening, for a check by justin before it lands. The
detailed designs come from two independent designs per problem, merged; the findings come from
`docs/reviews/red-team-57fa138.md`. Nothing below is pushed yet. Each item lands as its own
commit with tests, and only after brendan's OK.

| # | Item | Files (ours) | Touches justin's area? |
|---|---|---|---|
| 1 | Real mode's replicate phase proposes exactly `design.replication_specs(rows)["specs"]` and logs its ranking; draft `configs/agent/thresholds.yaml` copied from PREREG with section cites (himani to confirm) | loop, conductor, roles | Reads `design.replication_specs` only |
| 2 | RT-16: `Conductor._spent` charges the estimate for infra failures instead of $0 | conductor | No |
| 3 | RT-30: the Operator re-queues a `timeout` or `infra` failure once, through one shared constant | operator | No |
| 4 | RT-15/17: a single-writer `flock` helper, taken per `--out` in `loop.main` | new helper, loop | We propose that `launch_design` and `explore.main` call it too (you decide) |
| 5 | RT-32: one shared model-spend ledger for every entry point, re-read before each call | llm, loop, scout | Explore would share the same ledger and cap if it uses the factory (you decide) |
| 6 | Scout: Exa timeout (~30 s); keep partial search errors; one bad model item rejects only itself | search, agents | No |
| 7 | past.dev second reader: the Scout recalls as `scout`, drops statements an `analyst` wrote (the Scout never sees run results), and passes the rest to the model as quoted data | agents, context, cards/scout.md, ARCHITECTURE | Reads `PastMemory`; we don't edit pastdev.py |
| 8 | `.gitignore` local agent folders; `OPENAI_API_KEY` and `PAST_API_KEY` in `.env.example` (no new dependency, `uv run --with openai` stays) | .gitignore, .env.example | No |

After the confirmatory rows: restarting from HALTED (resume keeps the repeat counters), reading
`results/agent/veto.jsonl`, the rest of the Scout hardening, and one exploratory marker shared
with explore (RT-46).

## Questions for justin

1. Item 1: is `design.replication_specs` the right source for the Conductor's replicate phase,
   the same set `launch_design --only replication` launched?
2. Item 4: will `launch_design` and `explore.main` take the same lock, or do you want it
   another way?
3. Item 5: should explore's model calls share the loop's ledger and cap, or keep their own $10?
4. Item 7: is the Scout reading only `scout` and `explorer` statements right for your memory
   design? Should the Scout write its own claim-card data points, or does explore keep writing
   them?
5. Anything here that conflicts with what you are changing tonight?
