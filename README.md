# Green Infrastructure Optimizer

Optimize where to place green infrastructure (rain barrels, rain gardens, redeveloped
lots) across a simulated city grid so that the pollution carried by stormwater into
drains and local water bodies is minimized under a fixed budget.

This repo is the **local search + environment** half of the project. It's built so
that the richer Gymnasium environment and the cost-function features from the other
half can drop in without rewriting the agents.

## The problem

Rain washes pollutants (sediment, fertilizer nitrogen, tire rubber, etc.) off of impermeable urban
surfaces into stormwater systems and local water bodies, causing various environmental issues such as algae blooms, marine species die-off, and flooding.
Green spaces absorb and filter some of the volume from that runoff..
We model a city as an `m x n` grid:

- **infra** cells emit pollution
- water + pollution flows downstream along a flow field, accumulating from every
  source it passes
- **drain / water body** cells are sinks; pollution that reaches them (or runs off
  the grid edge) is the bad thing we minimize
- placing green cells along the flow path retains a fraction of the throughput

Green options, cheapest to priciest:

| type        | goes on   | cost | effect       |
|-------------|-----------|------|--------------|
| rain barrel | infra     | low  | small (15%)  |
| rain garden | empty     | mid  | medium (60%) |
| redevelop   | infra     | high | large (80%)  |

## Quick start

```bash
pip install -r requirements.txt        # numpy required; gymnasium/matplotlib optional
python -m examples.run_demo            # build a city, run all agents, compare, save default PNGs
python -m examples.run_demo --save board.png   # also write a picture
python -m pytest tests/                # smoke tests
python -m examples.run_adversary_study --seeds 0 1 --rows 6 --cols 6 --budget 100000 --attack-budget 1
python -m examples.run_adversary_study --seeds 0 1 --runoff-coeff 0.7 --storm-depth 2.0
```

Example run (12x12, budget 180000): steepest hill climbing cut delivered
pollution by **~75%** versus doing nothing on seed 7. The planned-layout demo also
runs when `--save` is omitted by assigning default output filenames.

The adversary study is a static robustness check: it first lets each placement
agent build a portfolio, then asks how much benefit remains when a small attack
budget disables GI assets or amplifies source emissions.

This copy also adds a lightweight storm/runoff layer: pollutant load can be
scaled by `runoff_coeff` and `storm_depth` before routing. A GI effectiveness map
lets static adversary runs reduce an installed asset's function without changing
the planner API or deleting the placement from the portfolio.

## Architecture

```
greenspace/
  grid.py          board layout, cell types, board generation, flow field, ascii render
  pollution.py     storm/runoff-scaled downstream simulation -> delivered pollution
  cost.py          CostConfig + objective()  (the modular cost function)
  env.py           GreenSpaceEnv: gymnasium interface + local-search helper API
  metrics.py       percent-reduction and robustness-ratio metrics
  adversary.py     static post-portfolio robustness evaluator
  agents/
    base.py            Agent interface, SolveResult, neighbor moves (add/remove/swap)
    hill_climbing.py   steepest + first-choice (with restarts), and constructive (spec-literal)
    simulated_annealing.py
    random_agent.py    baseline
examples/
  run_demo.py      compares all agents on one board
  run_adversary_study.py  paired-seed robustness experiment
  visualize.py     optional matplotlib render
tests/
  test_env.py
```

### The two interfaces on `GreenSpaceEnv` (this is the merge contract)

The env exposes **both** of these against the same state, on purpose:

1. **Standard Gymnasium** (`reset`, `step`, `action_space`, `observation_space`).
   Observation is a 6-channel `(C, m, n)` stack for grid state inspection.

2. **Local-search helpers** (`clone`, `place`, `remove`, `valid_placements`,
   `current_objective`, `baseline_pollution`). Local search evaluates whole
   candidate boards, so it uses these instead of crawling one `step` at a time.

Agents only ever touch the env API, never the board internals, so swapping in a
different environment doesn't break them as long as it keeps the same API.

### Where the other half plugs in (and how to extend it)

Three seams are built so new work doesn't touch core code:

- **New environment features → extra observation channels.** Attach an `(m, n)`
  array to `Board.extra` (e.g. `board.extra["slope"] = ...`) and it automatically
  becomes an extra channel in `env.observation()`. The observation space updates to
  match at construction.
- **New cost terms → `CostConfig.extra_terms`.** Append a callable taking a context
  dict (`pollution`, `spent`, `placements`, `board`, `env`) and returning a number;
  `objective()` sums it in. Example: a "disruption to daily life" penalty is one
  line, no edits to the objective function or the agents.
- **New agents → the `AGENTS` registry.** Implement `solve(env) -> SolveResult` and
  add one entry. The demo and any eval harness can iterate the registry.

- **gymnasium:** the env is a real `gymnasium.Env` when gymnasium is installed, and
  falls back to a tiny shim when it isn't, so it runs either way.

### Objective

`objective = pollution_weight * delivered_pollution + cost_weight * spent`
(lower is better). Default config uses `cost_weight = 0`, treating budget as a hard
cap and minimizing pollution. Set `cost_weight > 0` to trade spending against
pollution directly.

## Agents

- **Random** — baseline floor.
- **Constructive hill climbing** — the spec-literal search: neighbors are reached
  only by placing a green space, and it ends once the budget is fully utilized. Each
  step it commits the single placement that most reduces the objective.
- **Hill climbing** — steepest or first-choice over a larger add/remove/swap
  neighborhood, with random restarts to escape weak local optima.
- **Simulated annealing** — accepts worse moves with probability `exp(-delta/T)`,
  cooling geometrically; best explorer of the local search agents.

All return a `SolveResult` with final objective, pollution, spend, and the objective
history, so they're directly comparable and easy to plot. The `AGENTS` registry in
`greenspace/agents/__init__.py` lists them by name; adding an agent is one entry.

## Status / next steps

- [x] Gymnasium environment + flow model + modular cost function
- [x] Local search agents: constructive (spec-literal), hill climbing variants,
      simulated annealing, random baseline
- [x] Demo, visualization, smoke tests
- [x] Extension seams: extra observation channels, extra cost terms, agent registry
- [ ] Swap in the richer environment / feature set once it lands
