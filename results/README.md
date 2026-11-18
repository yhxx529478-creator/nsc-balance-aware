# Milestone preliminary results

These are compact summaries of the development experiments described in the draft.

| File | Evidence in the report |
|---|---|
| `ranking.csv` | All baselines, Spearman, Kendall tau-b, score ties and size-controlled comparisons |
| `lambda_grid.csv` | The complete initial, excess and global coefficient grids |
| `budget_selection.csv` | Oracle, best selected perplexity, regret and SD for every budget, k and method |
| `pairwise_changes.csv` | Corrections and harmful changes, including matched-size and near-score groups |
| `fold_results.csv` | Five-fold development coefficients and held-out-fold metrics |
| `preliminary_summary.json` | Initial six-pair diagnostics, synthetic controls, effective penalty strength and fold coverage |

CSV keys `v1_frozen` and `v2` retain the recorded identifiers for the adjacent and
excess penalties; they are not separate trained Transformer models. `global_tuned`
is the independently selected global CV-squared control.

Run `python -B reproduce.py` to regenerate all six summaries. Detailed scores,
selection-trial records and full pairwise diagnostics are written under `outputs/`
when the experiments run, rather than stored in this milestone upload. Synthetic
reorderings have no measured trained perplexity. No final outcomes are included.
