"""
data_replogle.py
----------------
Replogle 2022 K562 GWPS 벌크 데이터 로더.
HCEPerturbationPredictor 학습에 필요한 형식으로 변환.

데이터 구조
-----------
K562_gwps_raw_bulk.h5ad:
  obs: 각 행 = 섭동 조건 (index: {idx}_{GENENAME}_{guide}_{ensembl})
  var: 각 열 = 측정 유전자 (Ensembl ID, var['gene_name'] = symbol)
  X  : 벌크 평균 발현량 행렬 (11258 섭동 × 8248 유전자)
  obs['core_control']: True = 컨트롤 (non-targeting guide)

GO 라벨 생성 전략
-----------------
섭동된 유전자 → 사전정의된 경로 딕셔너리로 경로 귀속
delta expression이 큰 경로를 1로 레이블링
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# K562-relevant pathway gene sets (생물학적으로 검증된 핵심 유전자 목록)
# ======================================================================

# 각 리프 경로와 소속 유전자 목록 (HGNC symbol)
PATHWAY_GENES: Dict[str, List[str]] = {
    # ── 세포 주기 ───────────────────────────────────────────────────
    "g1s_transition": [
        "CCND1", "CCND2", "CCND3", "CDK4", "CDK6", "RB1", "RBL1", "RBL2",
        "E2F1", "E2F2", "E2F3", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B",
    ],
    "mitosis": [
        "CCNA2", "CCNB1", "CCNB2", "CDK1", "CDK2", "PCNA", "MCM2", "MCM3",
        "MCM4", "MCM5", "MCM6", "MCM7", "AURKA", "AURKB", "PLK1", "BUB1",
        "BUB3", "MAD2L1", "CDC20", "PTTG1",
    ],
    # ── DNA 복제/수선 ──────────────────────────────────────────────
    "dna_repair": [
        "BRCA1", "BRCA2", "RAD51", "PALB2", "FANCI", "FANCD2", "FANCA",
        "MLH1", "MSH2", "MSH6", "PMS2", "XRCC1", "XRCC4", "LIG4",
        "TP53", "CHEK1", "CHEK2", "ATM", "ATR", "PARP1",
    ],
    "dna_replication": [
        "PCNA", "RFC1", "RFC2", "RFC3", "RFC4", "RFC5", "POLA1", "POLD1",
        "POLE", "RPA1", "RPA2", "RPA3", "FEN1", "LIG1", "GINS1", "GINS2",
    ],
    # ── 전사 조절 ─────────────────────────────────────────────────
    "transcription_factors": [
        "MYC", "MYCN", "MYCL", "SP1", "KLF4", "KLF5", "GATA1", "GATA2",
        "TAL1", "SPI1", "FLI1", "NFE2", "RUNX1", "ETV6", "ERG",
    ],
    "chromatin_regulation": [
        "EZH2", "EZH1", "EED", "SUZ12", "DNMT1", "DNMT3A", "DNMT3B",
        "HDAC1", "HDAC2", "HDAC3", "KDM1A", "KDM5C", "KDM6A", "KMT2A",
        "KMT2D", "EP300", "CREBBP", "BRD4", "SMARCA4", "ARID1A",
    ],
    # ── 세포 사멸 ─────────────────────────────────────────────────
    "apoptosis": [
        "BCL2", "BCL2L1", "BCL2L2", "MCL1", "BAX", "BAK1", "BAD", "BIM",
        "PUMA", "NOXA", "CASP3", "CASP8", "CASP9", "CASP6", "CASP7",
        "DIABLO", "CYCS", "APAF1", "FAS", "FASLG", "TNFRSF10A",
    ],
    "necroptosis": [
        "RIPK1", "RIPK3", "MLKL", "TNFRSF1A", "TNFRSF1B", "FADD",
        "TRADD", "TRAF2", "TRAF5", "CIAP1", "CIAP2",
    ],
    # ── 신호 전달 ─────────────────────────────────────────────────
    "pi3k_akt_signaling": [
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PIK3R1", "PIK3R2",
        "AKT1", "AKT2", "AKT3", "PTEN", "MTOR", "RICTOR", "RAPTOR",
        "TSC1", "TSC2", "RHEB", "S6K1", "EIF4EBP1",
    ],
    "jak_stat_signaling": [
        "JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3",
        "STAT4", "STAT5A", "STAT5B", "STAT6", "SOCS1", "SOCS2", "SOCS3",
        "ABL1", "ABL2",
    ],
    # ── 리보솜/번역 ───────────────────────────────────────────────
    "ribosome_biogenesis": [
        "RPS14", "RPS15", "RPS19", "RPS24", "RPS26",
        "RPL5", "RPL10", "RPL11", "RPL22", "RPL26", "RPL35A",
        "NOG1", "NOB1", "WDR12", "BMS1", "EBNA1BP2",
    ],
    "translation": [
        "EIF4A1", "EIF4A2", "EIF4E", "EIF4G1", "EIF4G2",
        "EIF2S1", "EIF2S2", "EIF2S3", "EIF3A", "EIF3B",
        "EIF2AK1", "EIF2AK2", "EIF2AK3", "EIF2AK4",
    ],
    # ── 면역/인터페론 반응 ────────────────────────────────────────
    "interferon_signaling": [
        "IFNAR1", "IFNAR2", "IFNGR1", "IFNGR2",
        "MX1", "MX2", "OAS1", "OAS2", "OAS3", "OASL",
        "ISG15", "ISG20", "IFIT1", "IFIT2", "IFIT3",
        "IRF3", "IRF7", "IRF9",
    ],
    "cytokine_signaling": [
        "IL6", "IL6R", "IL6ST", "IL2", "IL2RA", "IL2RB", "IL2RG",
        "TNF", "TNFRSF1A", "IL1B", "IL1R1", "NFKB1", "NFKB2",
        "RELA", "RELB", "REL", "IKBKA", "IKBKB", "NEMO",
    ],
    # ── 단백질 품질 관리 ──────────────────────────────────────────
    "ubiquitin_proteasome": [
        "UBB", "UBC", "UBA1", "UBA2", "UBE2A", "UBE2B", "UBE2D1",
        "PSMA1", "PSMA2", "PSMA3", "PSMB1", "PSMB2", "PSMB3",
        "PSMD1", "PSMD2", "PSMD3", "CUL1", "CUL3", "FBXW7", "VHL",
    ],
    "autophagy": [
        "ATG1", "ATG5", "ATG7", "ATG12", "ATG16L1", "BECN1",
        "PIK3C3", "UVRAG", "LC3B", "SQSTM1", "NBR1", "TAX1BP1",
        "ULK1", "ULK2", "AMBRA1", "RUBCN",
    ],
}

# 상위 계층 구조 (ontology.py의 build_mock_go_perturbation_ontology와 연결)
# 여기서는 PATHWAY_GENES의 키(리프)를 GO 라벨 인덱스로 사용
LEAF_PATHWAYS = sorted(PATHWAY_GENES.keys())  # 알파벳 정렬 → 고정 인덱스
PATHWAY_TO_IDX = {p: i for i, p in enumerate(LEAF_PATHWAYS)}
N_PATHWAYS = len(LEAF_PATHWAYS)


def build_k562_go_ontology():
    """
    K562 분석에 맞는 GO 계층 구조 구성.
    PATHWAY_GENES의 리프를 사용하는 OntologyDAG 반환.
    """
    from HCE.ontology import OntologyDAG

    dag = OntologyDAG()

    # 중간 레벨 노드
    mid_nodes = {
        "cell_cycle":           ["g1s_transition", "mitosis"],
        "dna_process":          ["dna_repair", "dna_replication"],
        "gene_regulation":      ["transcription_factors", "chromatin_regulation"],
        "cell_death":           ["apoptosis", "necroptosis"],
        "signal_transduction":  ["pi3k_akt_signaling", "jak_stat_signaling"],
        "protein_synthesis":    ["ribosome_biogenesis", "translation"],
        "immune_response":      ["interferon_signaling", "cytokine_signaling"],
        "protein_quality":      ["ubiquitin_proteasome", "autophagy"],
    }

    # 최상위 레벨 노드
    top_nodes = {
        "cellular_process":   ["cell_cycle", "dna_process", "cell_death", "protein_quality"],
        "gene_expression":    ["gene_regulation", "protein_synthesis"],
        "signaling":          ["signal_transduction"],
        "immune_process":     ["immune_response"],
    }
    root = "biological_process"

    # 노드 추가
    dag.add_node(root, "Biological Process")
    for top in top_nodes:
        dag.add_node(top, top.replace("_", " ").title())
        dag.add_edge(top, root)
    for mid, leaves in mid_nodes.items():
        dag.add_node(mid, mid.replace("_", " ").title())
        for top, mids in top_nodes.items():
            if mid in mids:
                dag.add_edge(mid, top)
        for leaf in leaves:
            dag.add_node(leaf, leaf.replace("_", " ").title())
            dag.add_edge(leaf, mid)

    term_to_idx = {leaf: i for i, leaf in enumerate(LEAF_PATHWAYS)}
    return dag, term_to_idx


# ======================================================================
# Dataset
# ======================================================================

class ReplogleDataset(Dataset):
    """
    Replogle K562 벌크 섭동 데이터셋.

    각 샘플:
        expr         : (n_genes,) 섭동 후 발현량 (log1p 정규화됨)
        ctrl_expr    : (n_genes,) 컨트롤 평균 발현량
        delta_expr   : (n_genes,) 섭동 후 - 컨트롤
        pert_mask    : (n_genes,) {0,1} - 섭동된 유전자 위치
        go_labels    : (N_PATHWAYS,) {0,1} multi-label
        pert_gene    : str - 섭동 유전자명
    """

    def __init__(
        self,
        adata_path: str,
        gene_subset: Optional[int] = None,
        go_threshold_pct: float = 75.0,
    ):
        """
        Args
        ----
        adata_path:       h5ad 파일 경로
        gene_subset:      사용할 유전자 수 (None = 전체)
        go_threshold_pct: 경로별 |delta| 상위 몇 %를 활성으로 볼지
        """
        import anndata as ad

        print(f"[ReplogleDataset] 로딩: {adata_path}")
        adata = ad.read_h5ad(adata_path)

        # 섭동 유전자명 파싱 (index: {num}_{GENENAME}_{guide}_{ensembl})
        adata.obs["pert_gene"] = adata.obs_names.map(
            lambda x: x.split("_")[1]
        )

        # 컨트롤 평균 발현량
        ctrl_mask = adata.obs["core_control"].values
        ctrl_mean = adata.X[ctrl_mask].mean(axis=0)   # (n_genes,)

        # 측정 유전자 목록 (symbol)
        gene_names = adata.var["gene_name"].values.tolist()
        gene_set = set(gene_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # 유전자 서브셋 (메모리 절약)
        if gene_subset is not None and gene_subset < len(gene_names):
            # 분산이 큰 유전자 선택 (더 정보가 많음)
            var_per_gene = np.array(adata.X).var(axis=0) if not hasattr(adata.X, 'toarray') else adata.X.toarray().var(axis=0)
            top_idx = np.argsort(var_per_gene)[-gene_subset:]
            top_idx = np.sort(top_idx)
            adata = adata[:, top_idx]
            ctrl_mean = ctrl_mean[top_idx]
            gene_names = [gene_names[i] for i in top_idx]
            gene_to_idx = {g: i for i, g in enumerate(gene_names)}
            gene_set = set(gene_names)

        n_genes = len(gene_names)
        self.n_genes = n_genes
        self.gene_names = gene_names
        self.ctrl_mean = ctrl_mean.astype(np.float32)

        # 비컨트롤 섭동만 추출
        pert_adata = adata[~ctrl_mask]
        X = pert_adata.X.astype(np.float32)
        pert_genes = pert_adata.obs["pert_gene"].values

        print(f"  총 섭동 조건: {len(X)}, 유전자: {n_genes}")

        # ── GO 라벨 생성 ──────────────────────────────────────────
        # 방법: 각 섭동의 |delta_expr|을 각 경로 소속 유전자로 평균
        # 경로별 mean |delta|가 상위 go_threshold_pct% 이상 → 1

        delta = X - ctrl_mean[np.newaxis, :]          # (N_pert, n_genes)
        abs_delta = np.abs(delta)

        # 경로별 평균 |delta| 계산
        pathway_scores = np.zeros((len(X), N_PATHWAYS), dtype=np.float32)
        for pathway, idx in PATHWAY_TO_IDX.items():
            genes_in_pathway = [g for g in PATHWAY_GENES[pathway] if g in gene_to_idx]
            if not genes_in_pathway:
                continue
            gene_indices = [gene_to_idx[g] for g in genes_in_pathway]
            pathway_scores[:, idx] = abs_delta[:, gene_indices].mean(axis=1)

        # 각 경로마다 상위 pct 이상인 섭동을 1로
        thresholds = np.percentile(pathway_scores, go_threshold_pct, axis=0)
        go_labels = (pathway_scores >= thresholds[np.newaxis, :]).astype(np.float32)

        # 섭동 마스크 생성 (어떤 유전자가 섭동됐나)
        pert_mask = np.zeros((len(X), n_genes), dtype=np.float32)
        for i, gene in enumerate(pert_genes):
            if gene in gene_to_idx:
                pert_mask[i, gene_to_idx[gene]] = 1.0

        self.X = X
        self.delta = delta
        self.pert_mask = pert_mask
        self.go_labels = go_labels
        self.pert_genes = pert_genes

        print(f"  GO 라벨 양성 비율: {go_labels.mean():.3f}")
        print(f"  섭동 마스크 커버리지: {(pert_mask.sum(1) > 0).mean():.3f} (측정된 유전자 섭동 비율)")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "expr":       torch.tensor(self.X[idx]),
            "ctrl_expr":  torch.tensor(self.ctrl_mean),
            "delta_expr": torch.tensor(self.delta[idx]),
            "pert_mask":  torch.tensor(self.pert_mask[idx]),
            "go_labels":  torch.tensor(self.go_labels[idx]),
        }

    def get_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple["ReplogleDataset", "ReplogleDataset", "ReplogleDataset"]:
        """랜덤 split (train/val/test) 반환."""
        rng = np.random.default_rng(seed)
        n = len(self)
        idx = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return (
            _SubsetDataset(self, idx[:n_train]),
            _SubsetDataset(self, idx[n_train:n_train + n_val]),
            _SubsetDataset(self, idx[n_train + n_val:]),
        )


class _SubsetDataset(Dataset):
    def __init__(self, parent: ReplogleDataset, indices: np.ndarray):
        self.parent = parent
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.parent[self.indices[i]]
