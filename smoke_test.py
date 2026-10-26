"""Exercise the core model and loss without external data or checkpoints."""

import torch

from .model import HCEPerturbationPredictor
from .ontology import build_mock_go_perturbation_ontology


def main():
    torch.manual_seed(42)
    dag, terms = build_mock_go_perturbation_ontology()
    model = HCEPerturbationPredictor(
        n_genes=16, n_go_terms=len(terms), ontology=dag,
        go_term_to_idx=terms, hidden_dims=(32, 16), dropout=0.0,
    )
    expr = torch.randn(4, 16)
    mask = torch.zeros_like(expr)
    mask[:, 0] = 1
    delta = torch.randn_like(expr)
    labels = torch.randint(0, 2, (4, len(terms))).float()
    prediction, logits = model(expr, mask)
    assert prediction.shape == delta.shape
    assert logits.shape == labels.shape
    loss, _ = model.compute_loss(expr, mask, delta, labels)
    assert torch.isfinite(loss), "The combined loss must be finite."
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
    print(f"Core check passed: {len(dag)} ontology nodes, {len(terms)} leaves.")
    print(f"Prediction shape: {tuple(prediction.shape)}; loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
