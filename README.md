# HCE for single-cell perturbation models

Experiments with hierarchical cross-entropy (HCE) as an auxiliary objective for gene-expression prediction. The code connects expression models to pathway or cell-type hierarchies, with implementations for an MLP, GEARS, and scGPT.

The main question is whether learning the hierarchy helps prediction on held-out perturbations. A separate brain-cell analysis examines which input genes influence the classifier at different levels of its ontology.

## Start here

The repository installs as `HCE`, matching the imports used by the experiment scripts. Use Python 3.10 or newer in a virtual environment:

```bash
python -m pip install -e .
python -m HCE.smoke_test
```

The smoke test runs a forward and backward pass on synthetic tensors. It needs no dataset, pretrained model, GPU, or network connection after installation. It checks the core model and loss; it does not reproduce a benchmark.

## Model

Expression and a perturbation mask feed a shared representation, followed by two heads: one predicts expression changes, and the other predicts pathway labels.

[`ontology.py`](src/HCE/ontology.py) represents child-to-parent relationships. [`loss.py`](src/HCE/loss.py) propagates leaf labels through that hierarchy and combines the classification objective with expression regression. [`model.py`](src/HCE/model.py) contains the standalone MLP; the GEARS and scGPT modules attach related heads to their respective backbones.

## Recorded results

The K562 gene hold-out experiment records the following scores. These are saved runs, without a repeated-seed uncertainty estimate.

| Auxiliary weight | ID Pearson | OOD Pearson | ID pathway AUROC | OOD pathway AUROC |
| --- | ---: | ---: | ---: | ---: |
| 0.00 | 0.374 | 0.451 | 0.535 | 0.556 |
| 0.05 | 0.378 | 0.450 | 0.746 | 0.664 |
| 0.10 | 0.389 | 0.452 | 0.763 | 0.687 |
| 0.30 | 0.375 | 0.444 | 0.747 | 0.663 |

Source: [`ood_benchmark_v2.json`](results/ood_benchmark_v2.json). Pathway AUROC changes substantially in these runs; expression Pearson changes much less. At zero auxiliary weight, the pathway head has no classification training objective, so its AUROC is not a separately trained pathway baseline.

For Norman, the saved GEARS + HCE evaluation reports a mean test Pearson of **0.705 across 116 conditions**. Split-level results and the distinction between expression prediction and interaction prediction are documented in [the experiment notes](docs/experiments.md).

## Running experiments

Install the data-analysis dependencies for K562 experiments:

```bash
python -m pip install -e '.[analysis]'
export HCE_K562_DATA=/path/to/K562_gwps_raw_bulk.h5ad
python -m HCE.experiments.k562.benchmark_ood
```

GEARS and scGPT runs need their own working upstream environments, datasets, and checkpoints. Install this repository into those environments with `python -m pip install --no-deps -e .`. The [setup notes](docs/setup.md) list the entry points and path settings. Training data and model weights are not included.

## Repository guide

Source code lives in [`src/HCE/`](src/HCE/). The core classes remain available through `from HCE import ...`; dataset-specific runs have separate module paths.

| Directory | Contents |
| --- | --- |
| [`src/HCE/`](src/HCE/) | Ontology graph, losses, standalone models, path settings, and smoke test |
| [`data/`](src/HCE/data/) | Replogle dataset loader |
| [`ontologies/`](src/HCE/ontologies/) | GO and Hallmark builders and the bundled hierarchy |
| [`integrations/`](src/HCE/integrations/) | GEARS and scGPT wrappers |
| [`experiments/k562/`](src/HCE/experiments/k562/) | K562 demo and benchmarks |
| [`experiments/norman/`](src/HCE/experiments/norman/) | Norman training, split evaluation, and interaction analysis |
| [`experiments/brain/`](src/HCE/experiments/brain/) | Brain classifier, Jacobian analysis, and gene masking |
| [`results/`](results/), [`jacobian/`](jacobian/README.md) | Saved experiment records and figures |

The [code guide](docs/code-layout.md) gives the module boundaries and command changes. After updating from the earlier flat layout, rerun `python -m pip install -e .` in each environment that uses this repository.

The brain perturbation run has zero baseline RG classification accuracy on its evaluation sample. Its rankings are retained for diagnosis, but they do not establish genes that drive a cell-state transition. The saved monotonicity metric also makes no comparisons with internal parent nodes; see the experiment notes before interpreting that score.
