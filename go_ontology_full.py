"""
go_ontology_full.py
-------------------
실제 Gene Ontology(BP + CC + MF)를 obonet으로 파싱하여
HCE에 사용할 OntologyDAG + 유전자 목록 구성.

전략:
    1. go-basic.obo 로드 → 전체 DAG 구성
    2. 유전자 수 ≥ min_genes인 term만 필터링
    3. BP / CC / MF 세 가지 하위 온톨로지를 통합
    4. gene2go_all.pkl (GEARS) 로 gene → GO term 매핑

실행:
    python -m HCE.go_ontology_full
"""

from __future__ import annotations
import os
import sys
import pickle
import json
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import HCE.config as cfg

OBO_PATH    = cfg.GO_OBO
GENE2GO     = cfg.GENE2GO
RESULTS_DIR = cfg.RESULTS_ROOT

# GO namespace 약어
NS_MAP = {
    "biological_process":  "BP",
    "cellular_component":  "CC",
    "molecular_function":  "MF",
}


# ======================================================================
# OBO 파싱 + OntologyDAG 구성
# ======================================================================

def build_go_dag_from_obo(
    obo_path: str,
    gene2go_path: str,
    min_genes: int = 50,
    max_genes: int = 2000,
    namespaces: Optional[Set[str]] = None,
) -> Tuple["OntologyDAG", Dict[str, int], Dict[str, List[str]], Dict]:
    """
    OBO + gene2go → 필터링된 GO OntologyDAG.

    Parameters
    ----------
    min_genes:  term에 필요한 최소 유전자 수 (너무 작으면 노이즈)
    max_genes:  term의 최대 유전자 수 (너무 크면 비특이적)
    namespaces: 사용할 GO 하위 온톨로지 집합 (None = 전체)

    Returns
    -------
    dag:           OntologyDAG
    term_to_idx:   리프 term → 인덱스
    term_genes:    term_id → gene symbol 리스트
    meta:          통계 정보
    """
    import obonet
    from HCE.ontology import OntologyDAG

    if namespaces is None:
        namespaces = {"biological_process", "cellular_component", "molecular_function"}

    print(f"[GO DAG] OBO 로딩: {obo_path}")
    graph = obonet.read_obo(obo_path)
    print(f"  전체 노드: {graph.number_of_nodes()}, 엣지: {graph.number_of_edges()}")

    # ── gene2go 로딩 ──────────────────────────────────────────────
    with open(gene2go_path, "rb") as f:
        gene2go_raw: Dict[str, Set[str]] = pickle.load(f)

    # GO ID → 유전자 집합 반전
    go2genes: Dict[str, Set[str]] = defaultdict(set)
    for gene, go_ids in gene2go_raw.items():
        for go_id in go_ids:
            go2genes[go_id].add(gene)

    # ── 유효 term 필터링 ──────────────────────────────────────────
    # 조건: 올바른 namespace + 유전자 수 범위 + obsolete 아님
    valid_terms: Set[str] = set()
    for node_id, data in graph.nodes(data=True):
        ns = data.get("namespace", "")
        if ns not in namespaces:
            continue
        if data.get("is_obsolete"):
            continue
        n_genes = len(go2genes.get(node_id, set()))
        if min_genes <= n_genes <= max_genes:
            valid_terms.add(node_id)

    print(f"  필터링 후 유효 term: {len(valid_terms)}개 "
          f"({min_genes}≤유전자수≤{max_genes})")

    # ── OntologyDAG 구성 ──────────────────────────────────────────
    dag = OntologyDAG()

    # 세 namespace 루트 → 공통 루트
    dag.add_node("GO_root", "Gene Ontology Root")
    ns_roots = {
        "biological_process": "GO_BP",
        "cellular_component":  "GO_CC",
        "molecular_function":  "GO_MF",
    }
    for ns, ns_root in ns_roots.items():
        if ns in namespaces:
            dag.add_node(ns_root, NS_MAP[ns])
            dag.add_edge(ns_root, "GO_root")

    # 유효 term 추가 + is_a 엣지
    for node_id in valid_terms:
        data   = graph.nodes[node_id]
        ns     = data.get("namespace", "")
        name   = data.get("name", node_id)
        dag.add_node(node_id, name)

        # 부모 term
        parents_added = False
        for parent_id in graph.predecessors(node_id):   # obonet: is_a 방향
            if parent_id in valid_terms:
                dag.add_edge(node_id, parent_id)
                parents_added = True
        # 부모가 valid term에 없으면 namespace root에 연결
        if not parents_added and ns in ns_roots:
            dag.add_edge(node_id, ns_roots[ns])

    print(f"  DAG 노드: {len(dag.nodes)}, 엣지: {sum(len(v) for v in dag.parents.values())}")

    # ── 리프 term → 인덱스 ────────────────────────────────────────
    leaves = sorted(dag.get_leaves())
    # GO_root / namespace roots는 리프 제외
    pseudo_roots = {"GO_root", "GO_BP", "GO_CC", "GO_MF"}
    leaves = [l for l in leaves if l not in pseudo_roots]
    term_to_idx = {t: i for i, t in enumerate(leaves)}

    # ── term → gene list ─────────────────────────────────────────
    term_genes = {t: sorted(go2genes.get(t, [])) for t in valid_terms}

    meta = {
        "n_valid_terms":  len(valid_terms),
        "n_leaves":       len(leaves),
        "n_dag_nodes":    len(dag.nodes),
        "n_bp": sum(1 for t in valid_terms if graph.nodes[t].get("namespace") == "biological_process"),
        "n_cc": sum(1 for t in valid_terms if graph.nodes[t].get("namespace") == "cellular_component"),
        "n_mf": sum(1 for t in valid_terms if graph.nodes[t].get("namespace") == "molecular_function"),
    }

    return dag, term_to_idx, term_genes, meta


# ======================================================================
# 캐시 래퍼
# ======================================================================

def load_or_build_go_dag(
    min_genes: int = 50,
    max_genes: int = 2000,
    namespaces: Optional[Set[str]] = None,
    cache_dir: Optional[str] = None,
) -> Tuple["OntologyDAG", Dict[str, int], Dict[str, List[str]]]:
    """
    캐시가 있으면 로드, 없으면 OBO에서 빌드.
    """
    if cache_dir is None:
        cache_dir = cfg.GEARS_DATA_DIR
    ns_tag = "_".join(sorted(namespaces or {"BP", "CC", "MF"}))
    cache_file = os.path.join(
        cache_dir,
        f"go_dag_min{min_genes}_max{max_genes}_{ns_tag}.pkl"
    )

    if os.path.exists(cache_file):
        print(f"[GO DAG] 캐시 로드: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    dag, term_to_idx, term_genes, meta = build_go_dag_from_obo(
        OBO_PATH, GENE2GO, min_genes, max_genes, namespaces
    )

    print(f"[GO DAG] 캐시 저장: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump((dag, term_to_idx, term_genes), f)

    # 통계 JSON 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats_path = os.path.join(RESULTS_DIR, f"go_dag_stats_min{min_genes}.json")
    with open(stats_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  통계: {meta}")

    return dag, term_to_idx, term_genes


# ======================================================================
# ReplogleDataset - GO(BP+CC+MF) 버전
# ======================================================================

def build_full_go_dataset(
    adata_path: str,
    gene_subset: int = 2000,
    min_genes: int = 50,
    max_genes: int = 2000,
    go_threshold_pct: float = 75.0,
    cache_dir: Optional[str] = None,
):
    """
    BP + CC + MF 전체 GO 온톨로지 기반 ReplogleDataset.
    msigdb_ontology.build_hallmark_dataset()과 동일한 인터페이스.
    """
    import anndata as ad
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    dag, term_to_idx, term_genes = load_or_build_go_dag(
        min_genes=min_genes,
        max_genes=max_genes,
        cache_dir=cache_dir,
    )
    n_go = len(term_to_idx)

    print(f"\n[FullGO Dataset] 로딩: {adata_path}")
    adata = ad.read_h5ad(adata_path)
    adata.obs["pert_gene"] = adata.obs_names.map(lambda x: x.split("_")[1])

    ctrl_mask = adata.obs["core_control"].values
    ctrl_mean = adata.X[ctrl_mask].mean(axis=0).astype(np.float32)

    gene_names = adata.var["gene_name"].values.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    if gene_subset < len(gene_names):
        X_arr = adata.X.astype(np.float32)
        var_g  = X_arr.var(axis=0)
        top_idx = np.sort(np.argsort(var_g)[-gene_subset:])
        gene_names   = [gene_names[i] for i in top_idx]
        gene_to_idx  = {g: i for i, g in enumerate(gene_names)}
        ctrl_mean    = ctrl_mean[top_idx]
        adata_sub    = adata[:, top_idx]
    else:
        adata_sub = adata

    pert_adata = adata_sub[~ctrl_mask]
    X = pert_adata.X.astype(np.float32)
    pert_genes = pert_adata.obs["pert_gene"].values
    n_genes = len(gene_names)
    print(f"  샘플: {len(X)}, 유전자: {n_genes}, GO terms(leaf): {n_go}")

    delta     = X - ctrl_mean[np.newaxis, :]
    abs_delta = np.abs(delta)

    # GO 라벨: term_to_idx 기준 리프 term만
    leaves = sorted(term_to_idx.keys())
    pathway_scores = np.zeros((len(X), n_go), dtype=np.float32)
    for term_id, idx in term_to_idx.items():
        genes_in = [g for g in term_genes.get(term_id, []) if g in gene_to_idx]
        if genes_in:
            gidx = [gene_to_idx[g] for g in genes_in]
            pathway_scores[:, idx] = abs_delta[:, gidx].mean(axis=1)

    thresholds = np.percentile(pathway_scores, go_threshold_pct, axis=0)
    go_labels  = (pathway_scores >= thresholds[np.newaxis, :]).astype(np.float32)

    pert_mask = np.zeros((len(X), n_genes), dtype=np.float32)
    for i, gene in enumerate(pert_genes):
        if gene in gene_to_idx:
            pert_mask[i, gene_to_idx[gene]] = 1.0

    print(f"  GO 라벨 양성 비율: {go_labels.mean():.3f}")
    print(f"  섭동 마스크 커버리지: {(pert_mask.sum(1)>0).mean():.3f}")

    class FullGODataset(Dataset):
        def __init__(self):
            self.X = X; self.delta = delta; self.pert_mask = pert_mask
            self.go_labels = go_labels; self.ctrl_mean = ctrl_mean
            self.pert_genes = pert_genes; self.n_genes = n_genes
        def __len__(self): return len(self.X)
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
            n = len(self); idx = rng.permutation(n)
            n_train = int(n * train_ratio); n_val = int(n * val_ratio)
            return (Subset(self, idx[:n_train]),
                    Subset(self, idx[n_train:n_train+n_val]),
                    Subset(self, idx[n_train+n_val:]))

    return FullGODataset(), dag, term_to_idx


# ======================================================================
# 실행 검증
# ======================================================================

if __name__ == "__main__":
    # OBO 파일 다운로드 확인
    if not os.path.exists(OBO_PATH):
        print(f"[오류] OBO 파일 없음: {OBO_PATH}")
        print(f"다운로드: wget -P {cfg.GEARS_DATA_DIR}/ http://purl.obolibrary.org/obo/go/go-basic.obo")
        sys.exit(1)

    dag, term_to_idx, term_genes = load_or_build_go_dag(
        min_genes=50, max_genes=2000
    )
    print(f"\nDAG: {dag}")
    print(f"리프 term: {len(term_to_idx)}개")
    print(f"루트: {dag.get_roots()}")

    # 샘플 term 확인
    sample_terms = list(term_to_idx.keys())[:5]
    for t in sample_terms:
        anc = dag.get_ancestors(t, include_self=False)
        print(f"  {t} ({dag.nodes.get(t, '')}): depth={dag.get_depth(t)}, ancestors={len(anc)}")
