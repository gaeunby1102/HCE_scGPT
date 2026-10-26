# Brain-cell Jacobian analysis

This experiment attaches a hierarchical cell-type classifier to scGPT and measures the derivative of each ontology-node probability with respect to the model's input expression values.

The four output classes are RG, Neuroblast, excitatory neurons (`Ext`), and inhibitory neurons (`Inh`). The backbone is frozen in the recorded classifier-training script. Parent probabilities are sums over their descendant leaf probabilities.

```text
Cell
  Neural Cell
    Neuron
      Excitatory Neuron
      Inhibitory Neuron
    Neural Progenitor
      Radial Glia
      Neuroblast
```

The executable stages now live in [`src/HCE/experiments/brain/`](../src/HCE/experiments/brain/). This directory retains the saved analysis artifacts.

## Run order

Install the repository as `HCE` inside a working scGPT environment, then set the input paths as described in [setup](../docs/setup.md).

```bash
export HCE_SCGPT_BRAIN=/path/to/scGPT_brain
export HCE_BRAIN_ATLAS=/path/to/brain_atlas.h5ad
export HCE_JACOBIAN_RESULTS="$PWD/runs/brain/jacobian"

python -m HCE.experiments.brain.step1_finetune_hce
python -m HCE.experiments.brain.step2_hce_jacobian
python -m HCE.experiments.brain.step3_visualize
```

Step 1 saves `hce_brain_best.pt`; step 2 reads that checkpoint and saves `jacobian_results.json`; step 3 produces figures and an analysis report. The atlas and trained checkpoint are not included. The scripts expect the atlas fields and token vocabulary used by `BrainCellDataset`.

## Saved outputs

The committed [Jacobian results](jacobian_results.json) and [figures](figures/) can be inspected without training. The [original analysis report](HCE_Jacobian_Report.md) is retained as an experiment record. Its interpretation should be read alongside the following constraints.

- Sensitivity to an input value does not establish a gene's causal importance.
- A node spanning all four softmax classes has probability one, so its derivative is zero apart from numerical error. Root-level gradient comparisons therefore need different interpretation from leaf-level comparisons.
- Adding child probabilities does not guarantee monotonicity of absolute gradient magnitudes; derivatives can cancel.
- The separate masking run has a failed RG classification baseline, as recorded in the [experiment notes](../docs/experiments.md).

These outputs describe this trained classifier and its input representation. They are not a validated gene-regulatory network.
