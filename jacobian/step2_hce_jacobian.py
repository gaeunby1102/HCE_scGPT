"""
step2_hce_jacobian.py
---------------------
HCE fine-tuned scGPT_brain으로 Jacobian 계산.
∂P(ontology_node_k) / ∂(gene_expression_values) for all 8 brain cell ontology nodes.

실행:
    python -m HCE.jacobian.step2_hce_jacobian
"""
from __future__ import annotations
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import scanpy as sc
from collections import defaultdict

from HCE.jacobian.step1_finetune_hce import (
    build_brain_cell_ontology, load_scgpt_brain,
    BrainCellDataset, ScGPTBrainHCE,
    LEAF_TO_IDX, CELL_TYPES, N_CELLS, MAX_SEQ, N_BINS,
    DATA_PATH, SAVE_DIR,   # step1과 동일한 경로 재사용
)

CKPT_PATH  = os.path.join(SAVE_DIR, "hce_brain_best.pt")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
JAC_BATCH  = 8      # Jacobian 계산용 소배치 (메모리 절약)
N_JAC_CELLS = 200   # 세포 유형별 Jacobian 계산 세포 수

# Brain Cell Ontology 노드별 leaf 구성
# node_name → [leaf_indices] (softmax 4 출력 중 해당 노드의 자손 leaf)
NODE_LEAVES = {
    "Radial_Glia":        [LEAF_TO_IDX["RG"]],
    "Neuroblast_cell":    [LEAF_TO_IDX["Neuroblast"]],
    "Excitatory_Neuron":  [LEAF_TO_IDX["Ext"]],
    "Inhibitory_Neuron":  [LEAF_TO_IDX["Inh"]],
    "Neural_Progenitor":  [LEAF_TO_IDX["RG"], LEAF_TO_IDX["Neuroblast"]],
    "Neuron":             [LEAF_TO_IDX["Ext"], LEAF_TO_IDX["Inh"]],
    "Neural_Cell":        list(LEAF_TO_IDX.values()),
    "Cell":               list(LEAF_TO_IDX.values()),
}
ALL_NODES = list(NODE_LEAVES.keys())


def node_prob(leaf_probs: torch.Tensor, node: str) -> torch.Tensor:
    """leaf_probs (B, 4) → P(node) (B,)"""
    indices = NODE_LEAVES[node]
    return leaf_probs[:, indices].sum(dim=1)


def compute_jacobian_batch(model, gene_ids, values, pad_mask, vocab):
    """
    한 배치에서 모든 온톨로지 노드에 대한 Jacobian 계산.

    Returns
    -------
    grads: dict[node_name → np.array(B, L)]
        B개 세포 × L개 유전자 위치에서의 ∂P(node)/∂values
    gene_id_matrix: np.array(B, L)
        각 위치의 gene vocab ID
    """
    gene_ids = gene_ids.to(DEVICE)
    pad_mask = gene_ids.eq(vocab["<pad>"])

    # values를 requires_grad=True로 설정 (gradient 계산 대상)
    vals = values.to(DEVICE).float().requires_grad_(True)

    # forward (no_grad 없이 — gradient 흐름 필요)
    logits, _ = model(gene_ids, vals, pad_mask)
    leaf_probs = torch.softmax(logits, dim=-1)  # (B, 4)

    grads = {}
    for node in ALL_NODES:
        prob = node_prob(leaf_probs, node)  # (B,)
        # 배치 전체 합에 대한 gradient → 각 세포에 독립적 (단일 스칼라로 합산)
        if vals.grad is not None:
            vals.grad.zero_()
        g = torch.autograd.grad(
            prob.sum(), vals,
            retain_graph=True,
            create_graph=False,
        )[0]  # (B, L)
        grads[node] = g.detach().cpu().numpy()

    return grads, gene_ids.cpu().numpy()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_path = os.path.join(SAVE_DIR, "step2_jacobian.log")

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log("Step 2: HCE Jacobian 계산 (∂P(node)/∂gene_values)")
    log("=" * 60)

    # ── 온톨로지 & 모델 로드 ──────────────────────────────────────
    log("\n[1] 온톨로지 & 모델 로드...")
    dag, term_to_idx = build_brain_cell_ontology()

    scgpt_model, vocab, args = load_scgpt_brain()
    for p in scgpt_model.parameters():
        p.requires_grad_(False)

    model = ScGPTBrainHCE(scgpt_model, n_classes=4, d_model=args["embsize"])
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.cls_head.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()
    log(f"  체크포인트 로드: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    log(f"  분석 노드: {ALL_NODES}")

    # ── 데이터 준비 ───────────────────────────────────────────────
    log("\n[2] Fetal brain 데이터 준비...")
    adata = sc.read_h5ad(DATA_PATH)
    dataset = BrainCellDataset(adata, vocab, n_cells_per_type=N_JAC_CELLS,
                               max_seq=MAX_SEQ, n_bins=N_BINS)
    log(f"  총 {len(dataset)}개 세포 (유형별 최대 {N_JAC_CELLS})")

    # vocab id → gene symbol 역매핑
    idx_to_gene = {v: k for k, v in vocab.get_stoi().items()}

    # ── Jacobian 계산 ─────────────────────────────────────────────
    log(f"\n[3] Jacobian 계산 (batch={JAC_BATCH})...")
    log(f"  노드당 (n_cells × n_genes_per_cell) 행렬 축적")

    # cell_type별, node별 누적: {node: {cell_type: [abs_grad_per_gene_id]}}
    # gene_id (vocab int) → 누적 |grad| 및 count
    accum = {
        node: {ct: defaultdict(lambda: [0.0, 0]) for ct in CELL_TYPES}
        for node in ALL_NODES
    }

    idx_to_ct = {v: k for k, v in LEAF_TO_IDX.items()}

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=JAC_BATCH, shuffle=False)
    total = 0

    for batch_idx, (gene_ids, values, labels) in enumerate(loader):
        try:
            grads, gid_np = compute_jacobian_batch(model, gene_ids, values, None, vocab)
        except RuntimeError as e:
            log(f"  [경고] 배치 {batch_idx} 스킵: {e}")
            continue

        B = gene_ids.shape[0]
        total += B

        for b in range(B):
            ct_idx = labels[b].item()
            ct = idx_to_ct[ct_idx]
            g_ids = gid_np[b]   # (L,) gene vocab ids

            for node in ALL_NODES:
                abs_grad = np.abs(grads[node][b])  # (L,)
                for pos, (gid, ag) in enumerate(zip(g_ids, abs_grad)):
                    if gid == vocab["<pad>"]: continue
                    accum[node][ct][gid][0] += ag
                    accum[node][ct][gid][1] += 1

        if (batch_idx + 1) % 10 == 0:
            log(f"  배치 {batch_idx+1}/{len(loader)} 완료 ({total}개 세포)")

    log(f"\n  총 {total}개 세포 Jacobian 계산 완료")

    # ── 결과 정리 & 저장 ──────────────────────────────────────────
    log("\n[4] Top 유전자 추출 & 저장...")

    results = {}
    TOP_K = 200

    for node in ALL_NODES:
        results[node] = {}
        for ct in CELL_TYPES:
            d = accum[node][ct]
            if not d:
                results[node][ct] = {"top_genes": [], "top_scores": []}
                continue
            # gene_id → mean |grad|
            gene_scores = {
                idx_to_gene.get(gid, f"unk_{gid}"): cnt[0] / max(cnt[1], 1)
                for gid, cnt in d.items()
            }
            # Top-K 정렬
            top = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
            results[node][ct] = {
                "top_genes":  [g for g, s in top],
                "top_scores": [s for g, s in top],
            }

    # JSON 저장
    out_path = os.path.join(SAVE_DIR, "jacobian_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"  저장: {out_path}")

    # ── 주요 결과 출력 ────────────────────────────────────────────
    log("\n[5] 노드별 Top 10 마커 유전자")
    log("=" * 60)

    known_markers = {
        "RG":         ["PAX6", "SOX2", "HES1", "VIM", "NESTIN"],
        "Neuroblast": ["DCX", "TUBB3", "NEUROD1", "PROX1"],
        "Ext":        ["TBR1", "SATB2", "SLC17A7", "CUX1", "RELN"],
        "Inh":        ["GAD1", "GAD2", "DLX2", "SST", "PVALB"],
    }

    for node in ALL_NODES:
        log(f"\n  [{node}]")
        for ct in CELL_TYPES:
            top_genes = results[node][ct]["top_genes"][:10]
            markers = known_markers.get(ct, [])
            hits = [g for g in top_genes if g in markers]
            log(f"    {ct:12s} Top10: {top_genes}")
            if hits:
                log(f"    {'':12s} → known markers: {hits} ✓")

    # ── 계층 일관성 검증 ──────────────────────────────────────────
    log("\n[6] 계층 일관성: |∂parent| ≥ |∂child| 비율")
    child_parent_pairs = [
        ("Radial_Glia",       "Neural_Progenitor"),
        ("Neuroblast_cell",   "Neural_Progenitor"),
        ("Excitatory_Neuron", "Neuron"),
        ("Inhibitory_Neuron", "Neuron"),
        ("Neural_Progenitor", "Neural_Cell"),
        ("Neuron",            "Neural_Cell"),
        ("Neural_Cell",       "Cell"),
    ]
    for child, parent in child_parent_pairs:
        # 모든 세포 유형에 걸쳐 top gene overlap의 score 비율
        child_scores, parent_scores = [], []
        for ct in CELL_TYPES:
            c_genes = results[child][ct]["top_genes"][:50]
            p_genes = results[parent][ct]["top_genes"][:50]
            c_sc = dict(zip(results[child][ct]["top_genes"],
                            results[child][ct]["top_scores"]))
            p_sc = dict(zip(results[parent][ct]["top_genes"],
                            results[parent][ct]["top_scores"]))
            common = set(c_genes) & set(p_genes)
            for g in common:
                child_scores.append(c_sc[g])
                parent_scores.append(p_sc[g])
        if child_scores:
            monotone = np.mean(np.array(parent_scores) >= np.array(child_scores))
            log(f"  {child:22s} → {parent:18s}: {monotone:.3f} ({len(child_scores)} 유전자)")

    log(f"\nStep 2 완료. Step 3 시각화: python -m HCE.jacobian.step3_visualize")


if __name__ == "__main__":
    main()
