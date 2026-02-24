"""
demo.py
-------
HCE 모듈 데모: Replogle K562 실제 섭동 데이터로 전체 파이프라인 실행.

실행 (gears2 환경):
    /home/t1/miniconda3/envs/gears2/bin/python -m HCE.demo

기본 환경 (torch만 있는 경우):
    python -m HCE.demo
"""

from __future__ import annotations
import sys
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, "/data2/Atlas_Normal")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "/data2/Atlas_Normal/IL17RD_scdiffeq/jacobian_analysis/replogle_data/K562_gwps_raw_bulk.h5ad"
GENE_SUBSET = 2000   # 분산 상위 2000개 유전자만 사용 (메모리 절약)

print(f"[Device] {DEVICE}\n")


# ======================================================================
# Demo 1: 온톨로지 구조 검증
# ======================================================================

def demo_ontology():
    print("=" * 60)
    print("Demo 1: K562 GO 온톨로지 구조")
    print("=" * 60)

    from HCE.data_replogle import build_k562_go_ontology, LEAF_PATHWAYS, N_PATHWAYS
    dag, term_to_idx = build_k512_go_ontology()

    print(f"  {dag}")
    print(f"  루트: {dag.get_roots()}")
    print(f"  리프 경로 ({N_PATHWAYS}개): {LEAF_PATHWAYS}")
    print(f"  위상 정렬: {dag.topological_sort()}\n")

    # 경로별 조상 확인
    for leaf in ["apoptosis", "mitosis", "jak_stat_signaling"]:
        anc = dag.get_ancestors(leaf, include_self=False)
        print(f"  {leaf}: depth={dag.get_depth(leaf)}, ancestors={anc}")
    print()


# ======================================================================
# Demo 2: 실제 데이터 로딩 검증
# ======================================================================

def demo_data_loading():
    print("=" * 60)
    print("Demo 2: Replogle K562 데이터 로딩")
    print("=" * 60)

    from HCE.data_replogle import ReplogleDataset, N_PATHWAYS, LEAF_PATHWAYS

    dataset = ReplogleDataset(DATA_PATH, gene_subset=GENE_SUBSET)
    train_ds, val_ds, test_ds = dataset.get_splits(train_ratio=0.8, val_ratio=0.1)

    print(f"  전체: {len(dataset)}, train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # 샘플 확인
    sample = dataset[0]
    print(f"  샘플 키: {list(sample.keys())}")
    print(f"  expr shape: {sample['expr'].shape}")
    print(f"  go_labels shape: {sample['go_labels'].shape}")
    print(f"  pert_mask sum: {sample['pert_mask'].sum():.0f}")

    # GO 라벨 분포
    all_go = np.stack([dataset[i]["go_labels"].numpy() for i in range(min(1000, len(dataset)))])
    print(f"\n  GO 라벨 양성 비율 (처음 1000개 샘플):")
    for i, pname in enumerate(LEAF_PATHWAYS):
        print(f"    [{i:2d}] {pname:30s}: {all_go[:, i].mean():.3f}")
    print()
    return dataset


# ======================================================================
# Demo 3: HCEPerturbationPredictor 학습 (실제 K562 데이터)
# ======================================================================

def demo_hce_training(dataset=None):
    print("=" * 60)
    print("Demo 3: HCEPerturbationPredictor 학습 (K562 실제 데이터)")
    print("=" * 60)

    from HCE.data_replogle import ReplogleDataset, build_k562_go_ontology, N_PATHWAYS
    from HCE.model import HCEPerturbationPredictor

    if dataset is None:
        dataset = ReplogleDataset(DATA_PATH, gene_subset=GENE_SUBSET)

    train_ds, val_ds, test_ds = dataset.get_splits(train_ratio=0.8, val_ratio=0.1, seed=42)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    n_genes = dataset.n_genes
    dag, term_to_idx = build_k562_go_ontology()
    n_go = len(term_to_idx)

    print(f"  n_genes={n_genes}, n_go={n_go}, "
          f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model = HCEPerturbationPredictor(
        n_genes=n_genes,
        n_go_terms=n_go,
        ontology=dag,
        go_term_to_idx=term_to_idx,
        hidden_dims=(512, 256, 128),
        lambda_reg=1.0,
        lambda_cls=0.5,
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    best_val_loss = float("inf")
    best_state = None

    print("  학습 시작...")
    for epoch in range(20):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        ep_total = ep_reg = ep_cls = 0.0
        for batch in train_loader:
            expr      = batch["expr"].to(DEVICE)
            pert_mask = batch["pert_mask"].to(DEVICE)
            delta     = batch["delta_expr"].to(DEVICE)
            go_labels = batch["go_labels"].to(DEVICE)

            loss, info = model.compute_loss(expr, pert_mask, delta, go_labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_total += info["loss_total"]
            ep_reg   += info["loss_regression"]
            ep_cls   += info["loss_hier_cls"]

        scheduler.step()
        n_steps = len(train_loader)

        # ── Validation ─────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_pears  = []
        with torch.no_grad():
            for batch in val_loader:
                expr      = batch["expr"].to(DEVICE)
                pert_mask = batch["pert_mask"].to(DEVICE)
                delta     = batch["delta_expr"].to(DEVICE)
                go_labels = batch["go_labels"].to(DEVICE)

                _, info = model.compute_loss(expr, pert_mask, delta, go_labels)
                val_losses.append(info["loss_total"])

                # Pearson (발현 예측 품질)
                pred_delta, _ = model(expr, pert_mask)
                for i in range(len(expr)):
                    p = pred_delta[i].cpu().numpy()
                    t = delta[i].cpu().numpy()
                    if p.std() > 1e-6 and t.std() > 1e-6:
                        r = np.corrcoef(p, t)[0, 1]
                        val_pears.append(r)

        val_loss = np.mean(val_losses)
        val_pear = np.mean(val_pears) if val_pears else 0.0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"train_total={ep_total/n_steps:.4f} "
                  f"reg={ep_reg/n_steps:.4f} "
                  f"hier_cls={ep_cls/n_steps:.4f} | "
                  f"val_loss={val_loss:.4f} "
                  f"val_pearson={val_pear:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Test ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    test_pears = []
    test_go_aucs = []
    with torch.no_grad():
        for batch in test_loader:
            expr      = batch["expr"].to(DEVICE)
            pert_mask = batch["pert_mask"].to(DEVICE)
            delta     = batch["delta_expr"].to(DEVICE)
            go_labels = batch["go_labels"].to(DEVICE)

            pred_delta, go_logits = model(expr, pert_mask)
            go_probs = torch.sigmoid(go_logits).cpu().numpy()
            go_true  = go_labels.cpu().numpy()

            for i in range(len(expr)):
                p = pred_delta[i].cpu().numpy()
                t = delta[i].cpu().numpy()
                if p.std() > 1e-6 and t.std() > 1e-6:
                    test_pears.append(np.corrcoef(p, t)[0, 1])

            # GO AUROC
            for j in range(n_go):
                if go_true[:, j].sum() > 0 and go_true[:, j].sum() < len(go_true):
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(go_true[:, j], go_probs[:, j])
                    test_go_aucs.append(auc)

    print(f"\n  [최종 테스트 결과]")
    print(f"  Expression Pearson: {np.mean(test_pears):.4f} ± {np.std(test_pears):.4f}")
    print(f"  GO Classification AUROC: {np.mean(test_go_aucs):.4f} ± {np.std(test_go_aucs):.4f}")

    # 계층 예측 시각화 (첫 3 샘플)
    sample_batch = next(iter(test_loader))
    with torch.no_grad():
        results = model.predict_with_hierarchy(
            sample_batch["expr"][:3].to(DEVICE),
            sample_batch["pert_mask"][:3].to(DEVICE),
        )

    all_terms = list(dag.nodes.keys())
    all_probs = results["all_node_probs"]
    print(f"\n  [계층적 GO 예측 — 상위 5개 활성 노드]")
    for si in range(3):
        p = all_probs[si]
        top_k = p.topk(5)
        top_terms = [
            (all_terms[i.item()], v.item())
            for i, v in zip(top_k.indices, top_k.values)
        ]
        print(f"  Sample {si}: " + " → ".join(f"{t}({v:.3f})" for t, v in top_terms))

    print()
    return model


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    # Demo 1: 온톨로지
    from HCE.data_replogle import build_k562_go_ontology, LEAF_PATHWAYS, N_PATHWAYS
    print("=" * 60)
    print("Demo 1: K562 GO 온톨로지 구조")
    print("=" * 60)
    dag, term_to_idx = build_k562_go_ontology()
    print(f"  {dag}")
    print(f"  루트: {dag.get_roots()}")
    print(f"  리프 경로 ({N_PATHWAYS}개): {LEAF_PATHWAYS}")
    for leaf in ["apoptosis", "mitosis", "jak_stat_signaling"]:
        anc = dag.get_ancestors(leaf, include_self=False)
        print(f"  {leaf}: depth={dag.get_depth(leaf)}, ancestors={anc}")
    print()

    # Demo 2 + 3: 데이터 로딩 + 학습
    from HCE.data_replogle import ReplogleDataset
    print("=" * 60)
    print("Demo 2: Replogle K562 데이터 로딩")
    print("=" * 60)
    dataset = ReplogleDataset(DATA_PATH, gene_subset=GENE_SUBSET)
    sample = dataset[0]
    print(f"  전체 샘플: {len(dataset)}")
    print(f"  expr shape: {sample['expr'].shape}, go_labels shape: {sample['go_labels'].shape}")
    print()

    demo_hce_training(dataset)

    print("모든 데모 완료!")
