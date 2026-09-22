# Transport Demand and Economics

Biogeme multinomial logit analysis of travel-mode choice across car, public
transport, cycling, and walking. The repository includes the Python
implementation, input data, and coursework report.

## Setup

```bash
cd ~/dev/TDECoursework
uv sync
```

`uv sync` creates the project environment and installs runtime and development
dependencies, including Ruff and Pyright.

## Run

Run the complete report workflow:

```bash
uv run analysis
```

Run an individual scenario:

```bash
uv run analysis --scenario baseline
uv run analysis --scenario generalised
uv run analysis --scenario totalcost
uv run analysis --scenario intervention
```

Outputs are written to `outputs/`. The full workflow estimates the three model
specifications, performs likelihood-ratio tests, and generates market-share
plots for 400 public-transport intervention scenarios. Biogeme HTML, pickle,
and iteration artifacts are disabled.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run pyright
```
