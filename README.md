# NSC Balance Aware Architecture Ranking — Milestone Code

CS5491 project milestone · Anqi Xiang, Ruixi Wang and Hanyi Jiang.

This repository reproduces the preliminary experiments in our milestone report,
**Adjacent Layer Balance Aware Neural Spectral Capacity for Transformer Architecture
Ranking**. It is a development-stage submission, not the final project evaluation.

## Current progress

We implemented an adjacent-layer mismatch penalty for architecture-only Neural
Spectral Capacity (NSC), with a global CV-squared ablation. Its initial development
results contained three pairwise corrections and three harmful changes. A second
experiment subtracts the exact expected mismatch under random permutations of the
same layer capacities. It gives four corrections and one harmful change, a small
net improvement. Global imbalance and positional baselines remain competitive.

The repository contains both methods, controlled tests, development evaluation,
ablations and the compact result tables used in the draft. The scorers use no
weights, examples, gradients, architecture IDs or label lookups. Development labels
are used only for coefficient selection and evaluation. Final outcomes are absent.

## Structure

```text
README.md              Milestone scope, setup, commands and expected results
requirements.txt       Standard-library environment declaration
reproduce.py           Reproduce the draft's preliminary results
course_starter/        Unchanged course code and permitted development data
src/                   Adjacent and permutation-referenced scoring methods
scripts/               Evaluation, diagnostics and summary export
tests/                 14 adjacent/global tests and 9 excess-penalty tests
protocols/             Fixed grids, selection rules and diagnostic conventions
results/               Five summary tables and one supplementary result object
```

Start with `src/adjacent_score.py` and `src/excess_score.py` for the methods.
`scripts/run_adjacent.py` and `scripts/run_excess.py` implement the experiments.
`results/README.md` maps the included summaries to the report.

## Environment

Tested with Python 3.12.3 on Windows 11; the code requires Python 3.9+ and only the
standard library. No GPU, API key, model download, network access or training is needed.

An optional environment can be created from the repository root:

```text
python -m venv .venv
```

Activate `.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then:

```text
python -m pip install -r requirements.txt
```

The requirements file intentionally has no third-party packages.

## Reproduce the preliminary results

Run from the repository root:

```text
python -B reproduce.py
```

Expected final message: `PASS: milestone-development`.
The command runs both experiments, their controls and official-evaluator checks;
computes ranking changes, budget selection and development stability; and compares
all six compact summaries with the included results. Outputs go to a new
`outputs/run-TIMESTAMP/` directory:

- `summary/`: the six reproduced summaries and readable `milestone_tables.md`.
- `results/`: detailed scores, synthetic controls, fold records and selection trials.
- `validation.json` and step logs: execution status and measured runtime.

Detailed per-trial and full-pair exports are generated when needed rather than
included in this milestone upload. Source code and included summaries are unchanged.
To choose a new empty output directory:

```text
python -B reproduce.py --output outputs/my_run
```

To run only the 23 controlled tests and course sanity checks:

```text
python -B reproduce.py --check-only
```

Measured complete milestone run: 13.43 seconds on an Intel Core i7-1185G7, Windows 11 and Python 3.12.3. The generated metrics record
hardware and timings. Shared MP caches make scoring times unsuitable for fair
cold-start comparisons. No separate data-preparation, training or plotting step
is required: the small development archive is included and the draft uses tables.

## Evaluation and initial findings

The initial method uses `S0/(1+lambda*D_adj)`. The refined method uses
`S0/(1+lambda*max(0,D_adj-2L/(L-1)*CV^2))`, with zero excess for one or two layers
and all-zero capacities. Lambda zero recovers original NSC exactly. The random-order
expectation is exact; its use as a performance penalty is a heuristic.

Each penalty uses the grid `{0,0.1,0.25,0.5,1,2}`, maximizing development Spearman
against negative validation perplexity, preferring the smaller coefficient on exact
ties. Selected coefficients are 0.1 (adjacent), 0.5 (excess) and 1 (tuned global).
The fixed protocol files retain the selection rules and exploratory-analysis context.

| Method | Spearman rho | Kendall tau-b | Size-controlled tau-b |
|---|---:|---:|---:|
| Original NSC | 0.969745 | 0.861414 | 0.622222 |
| Adjacent penalty | 0.969877 | 0.861414 | 0.611111 |
| Excess penalty | 0.969925 | 0.862626 | 0.622222 |
| Tuned global | 0.970309 | 0.861818 | 0.622222 |
| Positional example | 0.970069 | 0.865051 | 0.622222 |

All comparisons use the same 100 development architectures. Size control gives 180
overlapping pairs at equal hidden width and min/max parameter ratio at least 0.90.
Budget experiments use 10M/20M/30M/50M total parameters, k=1/3 and 200 fixed-seed
trials, with uniform random selection as a baseline. Regret is best selected recorded
perplexity minus the feasible optimum. The adjacent and excess scores coincide
across all eight budget/k settings. All baselines and weaker outcomes are retained.

## Limitations and remaining work

The refinement was motivated after inspecting earlier development outcomes; its
small gains and near-score analyses are exploratory. Five-fold checks are development
stability analyses, and do not independently establish the sparse corrections.
Synthetic reorderings have no measured trained perplexities. Reduced effective
penalty strength and clipping remain competing explanations for the improvement.

Remaining work includes a matched-strength ablation, predefined repeated development
splits and final evaluation after the method is frozen. These are not claimed as
completed milestone results. This submission does not access final labels.

## Data and references

`course_starter/data/` includes 200 architecture specifications, the prescribed
100-development/100-final split, resource records and only the 100 development
validation labels. Memory, adaptive embeddings and `pre_lnorm=false` are preserved.
Upstream training seeds and the exact training revision are unavailable.
The course code is unchanged. Our safe checker skips only the final-label hash
in the original checker and retains all other numerical and permitted data checks.

- Zhu, Zhao and Lu. Neural Spectral Capacity: Measuring and Designing Architectures
  from Network Specification Alone. https://arxiv.org/abs/2609.23087
- Javaheripi et al. LiteTransformerSearch: Training-free Neural Architecture Search
  for Efficient Language Models. NeurIPS 2022. https://arxiv.org/abs/2203.02094
- Dai et al. Transformer-XL: Attentive Language Models beyond a Fixed-Length Context.
  ACL 2019. https://aclanthology.org/P19-1285/
