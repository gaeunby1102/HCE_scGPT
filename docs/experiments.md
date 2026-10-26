# Experiment records

The tables below summarize committed output files. They do not imply that all experiments used identical data, objectives, or evaluation metrics.

## K562 gene hold-out

[`ood_benchmark_v2.json`](../results/ood_benchmark_v2.json) contains the auxiliary-weight sweep shown in the main README. The target is an expression change, and pathway labels are derived from expression responses. The original and full-GO records are kept separately in [`ood_benchmark_results.json`](../results/ood_benchmark_results.json) and [`benchmark_full_go.json`](../results/benchmark_full_go.json).

The saved monotonicity score is 1.0 for every weight, including zero. The benchmark compares a leaf only with ancestors that also have output logits. With a leaf-only output mapping, there are no such comparisons, and the function returns 1.0. This metric does not measure consistency with internal parent nodes or isolate the effect of HCE training.

## Norman split evaluation

Source: [`eval_by_split.json`](../results/eval_by_split.json), GEARS + HCE with auxiliary weight 0.3.

| Test split | Conditions | Mean Pearson | Across-condition standard deviation |
| --- | ---: | ---: | ---: |
| `unseen_single` | 37 | 0.815 | 0.018 |
| `combo_seen0` | 9 | 0.739 | 0.069 |
| `combo_seen1` | 52 | 0.646 | 0.157 |
| `combo_seen2` | 18 | 0.631 | 0.163 |

The overall mean is 0.704856 across 116 conditions. Standard deviations describe variation across conditions, not uncertainty across training runs. The `seen0`, `seen1`, and `seen2` labels count how many genes in a double perturbation were seen during training.

The historical [`norman_comparison.json`](../results/norman_comparison.json) and training logs are also retained. Their labels and selected epochs are not sufficient for a clean matched baseline comparison, so they are not presented as evidence of a general improvement over GEARS.

## Double-perturbation interactions

[`interaction_analysis.py`](../src/HCE/experiments/norman/interaction_analysis.py) compares double-perturbation responses with an additive single-perturbation baseline. [`deep_interaction_analysis.py`](../src/HCE/experiments/norman/deep_interaction_analysis.py) examines individual cases and GO overlap. Their outputs are in [`interaction_analysis.json`](../results/interaction_analysis.json), [`deep_interaction_analysis.json`](../results/deep_interaction_analysis.json), and [`results/figures/`](../results/figures/).

Expression-profile correlation and interaction-residual correlation answer different questions. A model can score well on the former while missing the non-additive part of a response.

## Brain-cell analysis

The [Jacobian analysis](../jacobian/README.md) differentiates cell-ontology probabilities with respect to input values. A related [masking experiment](../results/brain_insilico_perturb.json) evaluates 300 genes in 200 RG cells. That run records baseline RG accuracy of 0.0, mean RG probability of 0.111, and mean Neuroblast probability of 0.172. Classifier behavior must be resolved before interpreting the resulting gene ranks as transition candidates.
