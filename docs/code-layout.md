# Code layout

```text
src/HCE/
  __init__.py
  config.py
  ontology.py
  loss.py
  model.py
  smoke_test.py
  data/
    replogle.py
  ontologies/
    go.py
    hallmark.py
    hallmark_ontology.json
  integrations/
    gears.py
    scgpt.py
  experiments/
    k562/
    norman/
    brain/
```

## Boundaries

The core graph, objectives, and MLP models do not import GEARS, scGPT, Scanpy, or experiment modules. The top-level `HCE` imports remain unchanged. `data/` handles dataset conversion, `ontologies/` builds external hierarchies, and `integrations/` contains wrappers for optional model dependencies.

Experiment modules own their training settings and evaluation procedures. K562 and Norman runs share the core objectives and loaders; the three brain attribution stages share the classifier definitions in `experiments/brain/step1_finetune_hce.py`.

Saved JSON, logs, and figures remain in the repository's `results/` and `jacobian/` directories. They are not bundled into the Python package. The Hallmark hierarchy is package data and stays beside its builder.

## Updated commands

Reinstall the editable package after moving from the previous layout:

```bash
python -m pip install --no-deps -e .
python -m HCE.smoke_test
```

| Earlier command or import | Current module |
| --- | --- |
| `HCE.demo` | `HCE.experiments.k562.demo` |
| `HCE.benchmark_ood` | `HCE.experiments.k562.benchmark_ood` |
| `HCE.benchmark_full_go` | `HCE.experiments.k562.benchmark_full_go` |
| `HCE.gears_norman_baseline` | `HCE.experiments.norman.gears_norman_baseline` |
| `HCE.gears_norman_hce` | `HCE.experiments.norman.gears_norman_hce` |
| `HCE.scgpt_norman_hce` | `HCE.experiments.norman.scgpt_norman_hce` |
| `HCE.scgpt_norman_finetune` | `HCE.experiments.norman.scgpt_norman_finetune` |
| `HCE.scgpt_gears_hce` | `HCE.experiments.norman.scgpt_gears_hce` |
| `HCE.eval_by_split` | `HCE.experiments.norman.eval_by_split` |
| `HCE.interaction_analysis` | `HCE.experiments.norman.interaction_analysis` |
| `HCE.deep_interaction_analysis` | `HCE.experiments.norman.deep_interaction_analysis` |
| `HCE.brain_insilico_perturb` | `HCE.experiments.brain.brain_insilico_perturb` |
| `HCE.jacobian.step1_finetune_hce` | `HCE.experiments.brain.step1_finetune_hce` |
| `HCE.jacobian.step2_hce_jacobian` | `HCE.experiments.brain.step2_hce_jacobian` |
| `HCE.jacobian.step3_visualize` | `HCE.experiments.brain.step3_visualize` |
| `HCE.data_replogle` | `HCE.data.replogle` |
| `HCE.go_ontology_full` | `HCE.ontologies.go` |
| `HCE.msigdb_ontology` | `HCE.ontologies.hallmark` |
| `HCE.gears_hce` | `HCE.integrations.gears` |
| `HCE.scgpt_hce` | `HCE.integrations.scgpt` |

Use `python -m <module>` for experiment entry points. The old module paths are not compatibility aliases. Public imports from `HCE`, `HCE.model`, `HCE.loss`, and `HCE.ontology` are preserved.

## Output paths

In an editable checkout, default outputs still go to repository `results/` and `jacobian/results/`. In a regular wheel installation, defaults are relative to the working directory. Set `HCE_RESULTS_ROOT` and `HCE_JACOBIAN_RESULTS` to select output locations explicitly. Moving the source package should never cause an experiment to write into `site-packages`.

The full list of data and checkpoint settings is in [setup](setup.md).
