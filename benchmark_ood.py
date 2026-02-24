"""
benchmark_ood.py  (v2)
-----------------------
HCE vs Baseline OOD 벤치마크 — 수정된 버전.

수정사항:
  1. lambda 스윕: 0.0, 0.05, 0.1, 0.3
  2. Monotonicity 계산 수정: sigmoid logit 기반으로 직접 계산
  3. OOD 분할: pert_mask가 0이 아닌 샘플만 (섭동 유전자가 측정된 샘플)
     → 진짜 OOD: 학습에서 한 번도 안 본 유전자 KO 샘플

실행:
    python -m HCE.benchmark_ood
"""

from __future__ import annotations
import os, sys, json, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

import HCE.config as cfg
from HCE.data_replogle import ReplogleDataset, build_k562_go_ontology, N_PATHWAYS
from HCE.model import HCEPerturbationPredictor

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH   = cfg.K562_DATA
RESULTS_DIR = cfg.RESULTS_ROOT
GENE_SUBSET = 2000
SEED        = 42


# ======================================================================
# Gene-OOD split (측정된 섭동 유전자만 대상)
# ======================================================================

def gene_ood_split(dataset: ReplogleDataset, ood_ratio=0.2, seed=42):
    """
    섭동 유전자가 측정 행렬에 포함된 샘플만 대상으로 OOD 분할.
    (pert_mask.sum > 0 인 샘플)
    """
    rng = np.random.default_rng(seed)
    pert_genes = np.array(dataset.pert_genes)
    pert_mask_sum = dataset.pert_mask.sum(axis=1)   # (N,)

    # 측정된 샘플만 (pert gene이 expression 행렬에 존재)
    measured_idx = np.where(pert_mask_sum > 0)[0]
    unmeasured_idx = np.where(pert_mask_sum == 0)[0]

    measured_genes = pert_genes[measured_idx]
    unique_genes   = np.unique(measured_genes)
    n_ood = max(1, int(len(unique_genes) * ood_ratio))
    ood_genes = set(rng.choice(unique_genes, n_ood, replace=False))

    is_ood = np.array([g in ood_genes for g in measured_genes])
    measured_train = measured_idx[~is_ood]
    ood_test       = measured_idx[is_ood]

    rng.shuffle(measured_train)
    split    = int(len(measured_train) * 0.85)
    train    = measured_train[:split]
    id_test  = measured_train[split:]

    # unmeasured 샘플도 학습에 포함 (섭동 결과 자체를 배우는 것도 도움)
    train_all = np.concatenate([train, unmeasured_idx])
    rng.shuffle(train_all)

    print(f"  Gene OOD split (측정된 샘플만 기준):")
    print(f"    고유 측정 유전자: {len(unique_genes)}, OOD 유전자: {n_ood}")
    print(f"    train: {len(train_all)}, id_test: {len(id_test)}, ood_test: {len(ood_test)}")
    return train_all, id_test, ood_test


# ======================================================================
# Monotonicity (수정버전: go_logits sigmoid 기반)
# ======================================================================

@torch.no_grad()
def compute_monotonicity(model, go_logits_all: torch.Tensor, term_to_idx, dag) -> float:
    """
    P(child) ≤ P(parent) 조건 충족 비율.
    ancestor_matrix를 쓰지 않고 dag.parents에서 직접 계산.
    """
    all_terms = list(dag.nodes.keys())
    term2i = {t: i for i, t in enumerate(all_terms)}
    go_probs = torch.sigmoid(go_logits_all)             # (N, n_leaf)

    violations = 0
    total = 0
    for leaf_term, leaf_idx in term_to_idx.items():
        ancestors = dag.get_ancestors(leaf_term, include_self=False)
        # 리프의 직계 부모 term 중 term_to_idx에 있는 것
        # (부모가 리프일 경우만 비교 가능 — 중간 노드는 별도 분류기 없음)
        parent_leaf = [a for a in ancestors if a in term_to_idx]
        for p in parent_leaf:
            p_idx = term_to_idx[p]
            viol = (go_probs[:, leaf_idx] > go_probs[:, p_idx] + 1e-4).sum().item()
            violations += viol
            total += len(go_probs)

    return 1.0 - violations / max(total, 1)


# ======================================================================
# 학습
# ======================================================================

def train_model(model, train_loader, epochs=25, lr=1e-3, label=""):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        ep_reg = ep_cls = 0.0
        for batch in train_loader:
            expr = batch["expr"].to(DEVICE)
            pert = batch["pert_mask"].to(DEVICE)
            dlt  = batch["delta_expr"].to(DEVICE)
            go   = batch["go_labels"].to(DEVICE)
            loss, info = model.compute_loss(expr, pert, dlt, go)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_reg += info["loss_regression"]
            ep_cls += info["loss_hier_cls"]
        sched.step()
        n = len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"  [{label}] ep{epoch+1:2d} reg={ep_reg/n:.4f} hier={ep_cls/n:.4f}")
    print(f"  [{label}] 완료 ({time.time()-t0:.1f}s)")
    return model


# ======================================================================
# 평가
# ======================================================================

@torch.no_grad()
def evaluate(model, loader, term_to_idx, dag, label, split):
    model.eval()
    pears, go_probs_all, go_true_all, logits_all = [], [], [], []

    for batch in loader:
        expr = batch["expr"].to(DEVICE)
        pert = batch["pert_mask"].to(DEVICE)
        dlt  = batch["delta_expr"].to(DEVICE)
        go   = batch["go_labels"].to(DEVICE)
        pred_delta, go_logits = model(expr, pert)
        go_probs_all.append(torch.sigmoid(go_logits).cpu())
        logits_all.append(go_logits.cpu())
        go_true_all.append(go.cpu())
        for i in range(len(expr)):
            p = pred_delta[i].cpu().numpy()
            t = dlt[i].cpu().numpy()
            if p.std() > 1e-6 and t.std() > 1e-6:
                pears.append(np.corrcoef(p, t)[0, 1])

    gp = torch.cat(go_probs_all).numpy()
    gl = torch.cat(logits_all)
    gt = torch.cat(go_true_all).numpy()

    # Pearson
    mean_pear = float(np.mean(pears)) if pears else 0.0
    std_pear  = float(np.std(pears)) if pears else 0.0

    # GO AUROC
    aucs = []
    for j in range(N_PATHWAYS):
        pos = gt[:, j].sum()
        if 0 < pos < len(gt):
            aucs.append(roc_auc_score(gt[:, j], gp[:, j]))
    mean_auc = float(np.mean(aucs)) if aucs else 0.0

    # Monotonicity (수정)
    mono = compute_monotonicity(model, gl, term_to_idx, dag)

    print(f"  [{label}] {split:8s} | "
          f"Pearson={mean_pear:.4f}±{std_pear:.4f} | "
          f"GO_AUROC={mean_auc:.4f} | Mono={mono:.4f}")

    return {"pearson": mean_pear, "pearson_std": std_pear,
            "go_auroc": mean_auc, "monotonicity": mono}


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 65)
    print("OOD Benchmark v2: lambda 스윕 + 수정된 Monotonicity")
    print("=" * 65)

    dataset = ReplogleDataset(DATA_PATH, gene_subset=GENE_SUBSET)
    dag, term_to_idx = build_k562_go_ontology()
    n_go = len(term_to_idx)

    train_idx, id_test_idx, ood_test_idx = gene_ood_split(dataset, ood_ratio=0.2, seed=SEED)

    train_loader    = DataLoader(Subset(dataset, train_idx),    batch_size=256, shuffle=True,  num_workers=4, pin_memory=True)
    id_test_loader  = DataLoader(Subset(dataset, id_test_idx),  batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    ood_test_loader = DataLoader(Subset(dataset, ood_test_idx), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # lambda 스윕
    lambdas = [0.0, 0.05, 0.1, 0.3]
    all_results = {}

    for lam in lambdas:
        name = f"λ={lam}"
        print(f"\n{'─'*65}\n  모델: HCE {name}\n{'─'*65}")
        model = HCEPerturbationPredictor(
            n_genes=dataset.n_genes, n_go_terms=n_go,
            ontology=dag, go_term_to_idx=term_to_idx,
            hidden_dims=(512, 256, 128),
            lambda_reg=1.0, lambda_cls=lam,
        ).to(DEVICE)
        model = train_model(model, train_loader, epochs=25, label=name)
        r = {}
        r["id"]  = evaluate(model, id_test_loader,  term_to_idx, dag, name, "ID")
        r["ood"] = evaluate(model, ood_test_loader, term_to_idx, dag, name, "OOD")
        all_results[name] = r

    # 요약 출력
    print(f"\n{'='*65}")
    print(f"{'모델':<10} {'분할':>6} {'Pearson':>10} {'GO_AUROC':>10} {'Mono':>8}")
    print(f"{'-'*65}")
    for name, r in all_results.items():
        for split, m in r.items():
            flag = "  ← OOD" if split == "ood" else ""
            print(f"{name:<10} {split:>6} {m['pearson']:>10.4f} "
                  f"{m['go_auroc']:>10.4f} {m['monotonicity']:>8.4f}{flag}")

    # 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/ood_benchmark_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
