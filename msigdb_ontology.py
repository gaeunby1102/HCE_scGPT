"""
msigdb_ontology.py
------------------
MSigDB Hallmark gene set 기반 GO 온톨로지 빌더.
hallmark_ontology.json의 DAG 구조 + gseapy로 실제 유전자 목록 로드.

data_replogle.py의 PATHWAY_GENES / build_k562_go_ontology()를 대체.

사용:
    from HCE.msigdb_ontology import build_hallmark_ontology, get_hallmark_pathway_genes
    dag, term_to_idx, pathway_genes = build_hallmark_ontology()
"""

from __future__ import annotations
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import HCE.config as cfg

ONTOLOGY_JSON = os.path.join(os.path.dirname(__file__), "hallmark_ontology.json")


def build_hallmark_ontology(
    cache_dir: Optional[str] = None,
) -> Tuple["OntologyDAG", Dict[str, int], Dict[str, List[str]]]:
    """
    MSigDB Hallmark 기반 OntologyDAG + 유전자 목록 반환.

    Returns
    -------
    dag:           OntologyDAG (52개 노드, 50개 리프)
    term_to_idx:   리프 노드 → 인덱스 (50개)
    pathway_genes: term_id → gene symbol 리스트
    """
    from HCE.ontology import load_ontology_from_json, OntologyDAG
    import gseapy as gp

    if cache_dir is None:
        cache_dir = cfg.GEARS_DATA_DIR

    # ── DAG 로드 ──────────────────────────────────────────────────
    dag = load_ontology_from_json(ONTOLOGY_JSON)
    leaves = sorted(dag.get_leaves())
    term_to_idx = {t: i for i, t in enumerate(leaves)}

    # ── MSigDB 유전자 목록 로드 (캐시) ────────────────────────────
    cache_path = cfg.HALLMARK_CACHE
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw_genes = json.load(f)
        print(f"[MSigDB] 캐시 로드: {cache_path}")
    else:
        print("[MSigDB] Hallmark gene sets 다운로드 중...")
        raw = gp.get_library("MSigDB_Hallmark_2020", organism="Human")
        raw_genes = {k: list(v) for k, v in raw.items()}
        with open(cache_path, "w") as f:
            json.dump(raw_genes, f, indent=2)
        print(f"[MSigDB] 저장: {cache_path}")

    # ── term_id → gene list 매핑 (JSON의 msigdb_key_map 활용) ────
    with open(ONTOLOGY_JSON) as f:
        meta = json.load(f)
    key_map: Dict[str, str] = meta["msigdb_key_map"]

    pathway_genes: Dict[str, List[str]] = {}
    for term_id, msigdb_key in key_map.items():
        if msigdb_key in raw_genes:
            pathway_genes[term_id] = raw_genes[msigdb_key]
        else:
            # 대소문자 무관 검색
            matches = [k for k in raw_genes if k.lower() == msigdb_key.lower()]
            pathway_genes[term_id] = raw_genes[matches[0]] if matches else []

    n_missing = sum(1 for g in pathway_genes.values() if not g)
    print(f"[MSigDB] 리프 {len(leaves)}개, 유전자 매핑 완료 (누락={n_missing})")
    return dag, term_to_idx, pathway_genes


def build_hallmark_dataset(
    adata_path: str,
    gene_subset: int = 2000,
    go_threshold_pct: float = 75.0,
    cache_dir: Optional[str] = None,
):
    """
    MSigDB Hallmark 기반 ReplogleDataset 생성 (data_replogle.py 대체).

    Returns
    -------
    dataset: ReplogleDataset-compatible 인스턴스
    dag:     OntologyDAG
    term_to_idx: Dict[str, int]
    """
    import anndata as ad
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    dag, term_to_idx, pathway_genes = build_hallmark_ontology(cache_dir)
    n_go = len(term_to_idx)
    leaves = sorted(dag.get_leaves())

    print(f"[HallmarkDataset] 로딩: {adata_path}")
    adata = ad.read_h5ad(adata_path)
    adata.obs["pert_gene"] = adata.obs_names.map(lambda x: x.split("_")[1])

    ctrl_mask = adata.obs["core_control"].values
    ctrl_mean = adata.X[ctrl_mask].mean(axis=0).astype(np.float32)

    gene_names = adata.var["gene_name"].values.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # 분산 상위 유전자 선택
    if gene_subset < len(gene_names):
        X_arr = adata.X.astype(np.float32)
        var_per_gene = X_arr.var(axis=0)
        top_idx = np.sort(np.argsort(var_per_gene)[-gene_subset:])
        gene_names = [gene_names[i] for i in top_idx]
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        ctrl_mean = ctrl_mean[top_idx]
        adata_sub = adata[:, top_idx]
    else:
        adata_sub = adata

    pert_adata = adata_sub[~ctrl_mask]
    X = pert_adata.X.astype(np.float32)
    pert_genes = pert_adata.obs["pert_gene"].values
    n_genes = len(gene_names)
    print(f"  샘플: {len(X)}, 유전자: {n_genes}, GO terms: {n_go}")

    delta = X - ctrl_mean[np.newaxis, :]
    abs_delta = np.abs(delta)

    # GO 라벨 (MSigDB 유전자 목록으로 경로 활성화 계산)
    pathway_scores = np.zeros((len(X), n_go), dtype=np.float32)
    for term_id, idx in term_to_idx.items():
        genes_in = [g for g in pathway_genes.get(term_id, []) if g in gene_to_idx]
        if genes_in:
            gidx = [gene_to_idx[g] for g in genes_in]
            pathway_scores[:, idx] = abs_delta[:, gidx].mean(axis=1)

    thresholds = np.percentile(pathway_scores, go_threshold_pct, axis=0)
    go_labels = (pathway_scores >= thresholds[np.newaxis, :]).astype(np.float32)

    # 섭동 마스크
    pert_mask = np.zeros((len(X), n_genes), dtype=np.float32)
    for i, gene in enumerate(pert_genes):
        if gene in gene_to_idx:
            pert_mask[i, gene_to_idx[gene]] = 1.0

    print(f"  GO 라벨 양성 비율: {go_labels.mean():.3f}")
    print(f"  섭동 마스크 커버리지: {(pert_mask.sum(1) > 0).mean():.3f}")

    class HallmarkReplogleDataset(Dataset):
        def __init__(self):
            self.X = X
            self.delta = delta
            self.pert_mask = pert_mask
            self.go_labels = go_labels
            self.ctrl_mean = ctrl_mean
            self.pert_genes = pert_genes
            self.n_genes = n_genes

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return {
                "expr":       torch.tensor(self.X[i]),
                "ctrl_expr":  torch.tensor(self.ctrl_mean),
                "delta_expr": torch.tensor(self.delta[i]),
                "pert_mask":  torch.tensor(self.pert_mask[i]),
                "go_labels":  torch.tensor(self.go_labels[i]),
            }

        def get_splits(self, train_ratio=0.8, val_ratio=0.1, seed=42):
            from torch.utils.data import Subset
            rng = np.random.default_rng(seed)
            n = len(self)
            idx = rng.permutation(n)
            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
            return (
                Subset(self, idx[:n_train]),
                Subset(self, idx[n_train:n_train + n_val]),
                Subset(self, idx[n_train + n_val:]),
            )

    return HallmarkReplogleDataset(), dag, term_to_idx


# ======================================================================
# CLI 검증
# ======================================================================
if __name__ == "__main__":
    import HCE.config as cfg
    dataset, dag, term_to_idx = build_hallmark_dataset(
        cfg.K562_DATA,
        gene_subset=2000,
    )
    print(f"\n온톨로지: {dag}")
    print(f"리프: {sorted(dag.get_leaves())}")
    print(f"샘플[0] go_labels sum: {dataset[0]['go_labels'].sum():.0f}")
