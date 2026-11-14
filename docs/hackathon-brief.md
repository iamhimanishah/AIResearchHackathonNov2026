# SF AI Research Hackathon: official brief

The event description as shared with participants on November 14, 2026.

**Removed from this copy because the repository is public:** credit codes and claim links,
passwords, the Wi-Fi details, Discord invite links and the venue's street address. All of them
are on the participant platform and in the event Discord. Each removal is marked in brackets.
Nothing else was changed.

---

## Welcome to the SF AI Research Hackathon

November 14-15, 2026 · San Francisco [venue name and street address removed]

- Discord: [invite link removed]. All news about the event will be posted there!
- Slides: [Opening slides](https://docs.google.com/presentation/d/1bRKP1LXw_3S8KOpP2w2cWhfu0BzhvlQ14TmCVRGLXiU/view)
- Wifi: [network name and password removed]

## API Keys & Credits

Please read this section carefully and claim your credits before hacking starts.

- Modal ($250): claim your credits: [claim link and code removed]
- Devin by Cognition ($200, 1 month of the Max plan): claim your code [claim link removed] with the email you used to apply. SWE-2 is free with this plan (CLI + Devin Desktop) if you run out of credits.
- Tinker by Thinking Machines ($100): your personal one-time code is in the API Keys section above.
- Exa ($50): sign up at [dashboard.exa.ai](https://dashboard.exa.ai/) and redeem the code [code removed].
- past.dev (500,000 credits per team): sign up at [past.dev](https://past.dev/), select 500,000 credits in the console and enter the code [code removed] (it only works with 500,000 selected), then create an API key. Questions: ask Lazare (past.dev) on site.
- OpenAI API credits: [claim link removed]

## Briefs & Docs

- Track 2 (Pacific): [full brief](https://drive.google.com/file/d/1r-dIO5Ol6RYLmcHXRwEUFr1vIBpi5CiZ/view) · [data pack (gpu-price-pack, about 300 MB)](https://drive.google.com/drive/folders/14zWi964QlLh3mZeWGrBI85J31-SqfGHN)
- Exa Challenge: [brief](https://drive.google.com/file/d/18wjMQsxnerTGfqOcsZMRkMvRNBkQ6Vyf/view) · [docs & quickstart](https://docs.exa.ai/)
- past.dev Challenge: [brief](https://drive.google.com/file/d/1uM3EgLx18NqHrTaSAKzIReK9YKTSludC/view) · [docs](https://past.dev/docs/memory-api/overview)

## Form your team

Teams of 3 to 5.

Already have a team? Go to the "Teams" panel on the platform, create your team and have your teammates join it.

Looking for a team? Join the Discord [invite link removed], introduce yourself in #intros-and-team-match and find your teammates :) You'll also have time at the kickoff to find teammates.

## Schedule

[Venue name and street address removed], San Francisco. All times PT.

Saturday, November 14: Kick off & build

- 09:00 AM Doors open, check-in & breakfast
- 09:30 AM Kickoff & sponsor challenge intros
- 10:00 AM Hacking begins
- 12:30 Lunch & sponsor office hours
- 07:00 PM Dinner, followed by overnight hacking
- 08:00 PM Reminder: your team must be on the platform by Sunday 12:00

Sunday, November 15: Judge & celebrate

- 08:00 AM Breakfast & final push
- 12:00 AM Lunch, team list closes on the platform
- 01:30 PM Judging schedule posted on Discord: your room and your time
- 02:00 PM Hacking ends, submissions due
- 02:45 PM Round 1: every team pitches in its room
- 04:45 PM Finals on stage
- 06:00 PM Prizes & closing ceremony, followed by drinks

## Tracks

Each team picks one track.

### Track 1: Build Your Own Autoresearch Framework

The question: can your system beat a best-known result in mathematics on its own?

What to build: a system that runs the research loop by itself: it picks what to try, runs it, reads the result and decides what to try next, until it beats a record. Karpathy's [autoresearch](https://rits.shanghai.nyu.edu/ai/karpathy-open-sources-autoresearch-100-ai-experiments-overnight-on-one-gpu/) (March 2026) showed the minimal version: an agent edits a single training script, runs 5-minute experiments and keeps only the changes that lower val_bpb. Take it further, for example with better search over ideas, memory of what failed, multi-objective evaluation, or a new domain.

Records to beat: any problem on EinsteinArena (public verifier + live leaderboard), any constant in Terence Tao's [table of optimization constants](https://teorth.github.io/optimizationproblems), or any of the 67 problems in the [AlphaEvolve repository of problems](https://github.com/google-deepmind/alphaevolve_repository_of_problems). Improving a known bound is welcome.

Judged on: how far past the record you get, how new your loop is, and whether we can follow each decision (show the log).

### Track 2: Recursive Self-Improvement by Pacific: An AI Scientist for GPU Value

The question: what will H100 and A100 GPUs resell for, and what will cloud rentals cost, in 2-3 years?

What to build: an AI scientist that works on this question the way a researcher would: it forms hypotheses about what drives GPU prices, tests them on real data, and improves its own forecasts with each iteration. Pick one direction:

- A. Self-improving forecasts: a system that learns from its previous predictions and new evidence (daily posted rental prices, GetDeploying rental price history).
- B. Market signals: forecast from market news, infrastructure developments and historical pricing (NVIDIA Newsroom, Epoch AI hardware data, BLS electricity distribution index). Explore how memory helps make sense of evolving evidence.
- C. Your own approach: bring your own question, signals or methods.

Data: Pacific's data pack (gpu-price-pack, data cutoff 6 November 2026, CC BY 4.0) with 3.7M+ posted rental offers, price indices and hardware data, plus a starter kit that builds price series, runs baselines and scores forecasts. [Download the data pack](https://drive.google.com/drive/folders/14zWi964QlLh3mZeWGrBI85J31-SqfGHN).

How it's scored: forecasts start from 6 November at horizons of 1, 3 and 7 days, scored on Sunday against posted rental prices collected after the cutoff (pinball loss at quantiles 0.1, 0.5 and 0.9, skill vs. the last-value baseline). The 2-3 year view can't be scored by Sunday: argue it from public evidence, with your assumptions and ranges.

Submit: forecasts in the starter kit format, one command that reproduces them, and concise findings (question, method, evaluation, results with uncertainty, what your system learned across iterations).

Judged on: rigour, useful economic insight, reproducibility and originality. All details in the [full brief](https://drive.google.com/file/d/1r-dIO5Ol6RYLmcHXRwEUFr1vIBpi5CiZ/view).

### Track 3: AI Scientist Swarms

The question: can a team of agents reproduce a full computational biology study from raw data better than the best single frontier model?

What to build: generating hypotheses is the easy part, rejecting the wrong ones is harder. Build a multi-agent system where agents with different roles (proposer, experimenter, critic) plan, run the analysis and check each other's results before anything is reported.

Record to beat: BixBench3 (Edison Scientific), 20 published studies rebuilt from raw data. Best score: 0.523.

Context: [Kosmos](https://arxiv.org/html/2511.02824v2) (Edison Scientific, 2025) coordinates 200+ agents through a shared world model, and independent scientists judged 79.4% of its statements accurate, so about 1 in 5 is wrong. [Popper](https://arxiv.org/abs/2502.09858) (Huang et al., ICML 2025) has agents design falsification experiments with Type-I error control.

Judged on: score vs. the record and a single-agent baseline on the same compute budget, rigour, reproducibility. Report what the swarm rejected as well as what it found.

### Track 4: World Models

The question: how long can a generated world hold together while an agent plays in it?

What to build: start from an open world model (e.g. Matrix-Game 3.0 or ABot-World-0) and make it hold together longer: memory, state tracking, re-anchoring, fine-tuning. No training from scratch.

Record to beat: PlayWorld. Genie 3 (closed) leads at 2.12, Matrix-Game 3.0 scores 1.14. Aim for an open model that beats Genie 3.

Context: [Dreamer 4](https://arxiv.org/abs/2509.24527v1) (Hafner, Yan, Lillicrap, 2025) trained an agent entirely inside a world model learned from offline video.

Judged on: PlayWorld score vs. the record and your starting model, the method, and rigour (show where your world still breaks).

### Track 5: Open Track

Your idea doesn't fit tracks 1–4? Build whatever you want. Same rules: built during the hackathon, public repo, demo, same judging criteria. Sponsor challenges (Exa, past.dev, Modal) are open to this track too.

## Sponsor Challenges

Optional and open to every track: enter as many as you like, on top of your track.

### Exa: Research agents that search

The prompt: build a research agent, AI scientist or self-improving system that uses Exa to find and work with information from the web. The strongest projects do something they couldn't do from model knowledge alone: pull in data published last week, check a claim against real sources, or discover a method nobody told the agent about.

Requirements:

- At least one Exa API call (search, contents, find similar, answer, research or Websets) runs in your project's main loop, not only in a side feature.
- Submit a repo link and point to where Exa is called.
- Show your sources: the demo makes clear what the system found through Exa and how it used it.

Judged on: Exa is central (30%), research value (25%), creativity (25%), checkable output (20%).

Prizes: $5K / $3K / $1K in Exa credits + Exa hoodies, Exa-branded TRMNLs and Exa goats. [Full brief](https://drive.google.com/file/d/18wjMQsxnerTGfqOcsZMRkMvRNBkQ6Vyf/view) · [docs](https://docs.exa.ai/)

### past.dev: Multi-Agent Memory Challenge

The prompt: past.dev is a multiplayer long-term memory API for agents. Add a shared memory layer to your main project: your agents write facts and data points to past.dev and use the ones written by other agents.

Requirements:

- More than one agent writes to the memory layer.
- More than one agent reads from it.
- At least one agent reads data written by a different agent and uses it in its tasks.

You can combine past.dev with other memory techniques (markdown, RAG): say in your README which one holds what. You don't have to prove the memory improves your results: if it didn't, explain why in your README.

Judge: Mehdi Djabri (CEO, past.dev). Mentor on site: Lazare Masset.

Prize: a Bambu Lab P1S 3D printer for each team member + $1K in past.dev credits. [Full brief](https://drive.google.com/file/d/1uM3EgLx18NqHrTaSAKzIReK9YKTSludC/view) · [docs](https://past.dev/docs/memory-api/overview)

### Best use of Modal

The best use of Modal for training, inference, or running large numbers of experiments and agent sandboxes in parallel. Open to any team using Modal.

Prize: $5,000 in Modal credits.

### Most frontier agentic work by Originator

Hack directly on [hackwithori.com](http://hackwithori.com/) ([password removed]) with 24/7h cloud based agents (codex, claude code..) or upload your sessions when done to participate to the challenge.

An agent will review your work & pre-score it for the judges.

Prize: $1,000 in cash for #1st, $500 for #2nd, $500 for #3rd

Not a finalist? The winner still demos on stage (2 min).

## Judging

5 criteria, equal weight (20% each):

- Research rigour: clear question, sound experiment design, baselines or ablations, claims backed by results
- Novelty: a new idea, method or use case, beyond existing papers and open-source baselines
- Results: a measurable result on a real problem or benchmark, ideally against a baseline; a working live demo
- Autonomy & self-improvement: how much of the research loop runs without a human; extra credit if it gets better from its own results
- Interpretability & reproducibility: decisions can be followed (logs, traces, sources), code runs, failures reported honestly

What to submit (by Sunday 14:30)

- 1 track, plus any challenges you enter
- The link to your GitHub repo
- A 2-minute demo video (YouTube, Drive or Loom)
- A short description of your project

How it works

- Round 1 (5 min per team, 6 rooms): pitch (3:00) + Q&A (2:00). The top 2 of each track go to the finals.
- Finals at 16:30 (3 min per team, on stage): pitch + live demo, no Q&A. Then the challenge winners demo (2 min).
- The final panel picks the top 3. All winners are announced at 18:00.

## Prizes

Top 3 overall (all tracks combined)

- 1st: $5K OpenAI credits + $5K Tinker credits + cash prize + Boardy Pro
- 2nd: $2.5K OpenAI credits + cash prize + Boardy Pro
- 3rd: $1K OpenAI credits + cash prize + Boardy Pro

Tracks & challenges (stack on top of the top 3)

- Track 2 (Pacific): $5K in compute ($3K / $1.5K / $500)
- Exa: $5K / $3K / $1K in credits + hoodies, TRMNLs and more
- Modal: $5,000 in credits
- past.dev: a Bambu Lab P1S 3D printer for each team member + $1K in credits
- Best photo on X: $500 cash

## Rules

- Build entirely during the event. No prior commits to the repo.
- Teams of 3 to 5.
- Open-source libraries, APIs and pre-trained models are allowed if properly credited.
- Submit all code and demos before the deadline to be eligible.
- Follow the law, ethical AI practices and the Code of Conduct. Never use data without consent.
- Use partner and sponsor APIs according to their terms of service.
- All team members must be physically present at the venue. Remote participation is not permitted.
- Organizers may disqualify any team for rule violations or unfair play.

## Code of Conduct

- Treat everyone with courtesy, whatever their background or experience.
- Zero tolerance for demeaning, threatening or discriminatory behavior.
- Compete fairly: help, share, don't sabotage.
- Respect the venue and equipment, and clean up after yourself.
- Wellbeing over winning: rest, eat and check on your teammates.
- If something feels wrong, tell an organizer. Reports stay confidential.
