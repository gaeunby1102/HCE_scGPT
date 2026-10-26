# Experiment setup

Run commands from the repository root after installing the `HCE` package. The core smoke test uses only NumPy and PyTorch. The optional `analysis` dependencies cover data loading, ontology construction, and plotting. They are dependency declarations, not a lockfile for the original training environment.

The original experiment notes record Python 3.11, PyTorch 2.3.0, PyTorch Geometric 2.5.3, and Scanpy 1.11.5. GEARS also needs its graph extensions; scGPT needs its tokenizer and model dependencies. Keep those upstream environments separate where their requirements conflict.

## Data and checkpoints

[`config.py`](../config.py) reads these environment variables when imported:

| Variable | Default | Used for |
| --- | --- | --- |
| `HCE_DATA_ROOT` | `./data` | Base directory for external inputs |
| `HCE_K562_DATA` | `$HCE_DATA_ROOT/K562_gwps_raw_bulk.h5ad` | K562 perturbation data |
| `HCE_GEARS_DATA` | `$HCE_DATA_ROOT/gears` | GEARS datasets and annotations |
| `HCE_GEARS_SAVE_NORMAN` | `$HCE_DATA_ROOT/gears_hce_norman` | GEARS + HCE checkpoint directory |
| `HCE_GEARS_SAVE_BASELINE` | `$HCE_DATA_ROOT/gears_baseline_norman` | Baseline checkpoint directory |
| `HCE_SCGPT_BRAIN` | `$HCE_DATA_ROOT/scGPT_brain` | Pretrained scGPT directory |
| `HCE_BRAIN_ATLAS` | `$HCE_DATA_ROOT/brain_atlas.h5ad` | Brain atlas |
| `HCE_RESULTS_ROOT` | Repository `results/` | Experiment outputs |
| `HCE_JACOBIAN_RESULTS` | Repository `jacobian/results/` | Brain classifier and attribution outputs |

GO resources can be set with `HCE_GO_OBO`, `HCE_GENE2GO`, `HCE_GO_CACHE`, and `HCE_HALLMARK_CACHE`. The Hallmark builder may download gene sets when its cache is absent. Inputs must match the schemas expected by the loaders; setting a path alone does not prepare a dataset.

To keep a new run separate from the committed results:

```bash
export HCE_DATA_ROOT=/path/to/data
export HCE_RESULTS_ROOT="$PWD/runs/experiment-01"
export HCE_JACOBIAN_RESULTS="$PWD/runs/experiment-01/jacobian"
mkdir -p "$HCE_RESULTS_ROOT" "$HCE_JACOBIAN_RESULTS"
```

## Entry points

| Command | Prerequisites |
| --- | --- |
| `python -m HCE.smoke_test` | Core installation only; synthetic input |
| `python -m HCE.benchmark_ood` | K562 data and analysis dependencies |
| `python -m HCE.benchmark_full_go` | K562 data and GO resources |
| `python -m HCE.gears_norman_baseline` | GEARS environment and Norman data |
| `python -m HCE.gears_norman_hce` | GEARS environment and Norman data |
| `python -m HCE.eval_by_split` | Saved GEARS + HCE model and its data split |
| `python -m HCE.interaction_analysis` | Saved GEARS + HCE model and Norman data |
| `python -m HCE.deep_interaction_analysis` | The preceding interaction analysis outputs |
| `python -m HCE.scgpt_norman_hce` | scGPT checkpoint and Norman data |
| `python -m HCE.scgpt_norman_finetune` | scGPT checkpoint and Norman data |
| `python -m HCE.scgpt_gears_hce` | GEARS and scGPT dependencies and inputs |
| `python -m HCE.brain_insilico_perturb` | Brain atlas, pretrained scGPT, trained HCE brain head |

These are experiment scripts with settings near the top of each module, not a single configurable training application. Inspect the settings before a training run. The scripts and saved outputs are available here; the original data, checkpoints, and complete environment snapshots are not.
