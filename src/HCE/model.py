"""
model.py
--------
HCE를 통합한 두 가지 모델:

1. HCECellTypeClassifier — 세포 유형 분류기 (논문 직접 구현체)
2. HCEPerturbationPredictor — GEARS 스타일 섭동 예측기 + HCE
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import HierarchicalCrossEntropyLoss, HierarchicalPerturbationLoss
from .ontology import OntologyDAG


# ======================================================================
# 1. 세포 유형 분류기 (Cell Type Classifier with HCE)
# ======================================================================

class HCECellTypeClassifier(nn.Module):
    """
    싱글셀 RNA-seq → 세포 유형 계층 분류기.
    표준 CE 대신 HCE 손실함수를 사용하여 OOD 강건성 확보.

    Architecture: Gene expression → MLP encoder → class logits
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        ontology: OntologyDAG,
        term_to_idx: Dict[str, int],
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
        hce_alpha: float = 0.7,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_classes = n_classes

        # MLP 인코더
        layers = []
        in_dim = n_genes
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

        # HCE 손실함수
        self.criterion = HierarchicalCrossEntropyLoss(
            ontology=ontology,
            term_to_idx=term_to_idx,
            alpha=hce_alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_genes) 유전자 발현 행렬
        반환: (B, n_classes) 로짓
        """
        h = self.encoder(x)
        return self.classifier(h)

    def compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = self(x)
        return self.criterion(logits, labels)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """예측 라벨 또는 확률 반환."""
        logits = self(x)
        probs = F.softmax(logits, dim=-1)
        if return_probs:
            return probs
        return probs.argmax(dim=-1)


# ======================================================================
# 2. 섭동 예측기 (GEARS-style + HCE)
# ======================================================================

class HCEPerturbationPredictor(nn.Module):
    """
    유전자 섭동 결과 예측기 with HCE.

    GEARS의 핵심 아이디어(유전자 관계 그래프 + 섭동 임베딩)를
    단순화하면서, HCE를 통해 예측 결과가 GO 계층 구조를 만족하도록 학습.

    입력:
        - 기준 세포 발현 프로파일 x ∈ R^G
        - 섭동 유전자 마스크 p ∈ {0,1}^G (1 = knock-out 대상)

    출력:
        - 예측 섭동 후 발현량 Δx ∈ R^G  (변화량)
        - GO 프로세스 영향 로짓 z ∈ R^K  (어떤 경로가 영향받는지)

    Architecture
    ------------
    [x; p]  →  Expression Encoder
           →  Perturbation Head   →  Δx (regression)
           →  GO Classifier Head  →  z  (hierarchical classification)
    """

    def __init__(
        self,
        n_genes: int,
        n_go_terms: int,
        ontology: OntologyDAG,
        go_term_to_idx: Dict[str, int],
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        dropout: float = 0.1,
        lambda_reg: float = 1.0,
        lambda_cls: float = 0.5,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_go = n_go_terms

        # 공유 인코더: 입력 = 발현량(G) + 섭동 마스크(G) = 2G
        layers = []
        in_dim = n_genes * 2
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.latent_dim = in_dim

        # ① 발현 변화량 예측 헤드 (회귀)
        self.expr_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, n_genes),
        )

        # ② GO 프로세스 분류 헤드 (계층 분류)
        self.go_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.GELU(),
            nn.Linear(self.latent_dim // 2, n_go_terms),
        )

        # 손실함수
        self.criterion = HierarchicalPerturbationLoss(
            ontology=ontology,
            go_term_to_idx=go_term_to_idx,
            lambda_reg=lambda_reg,
            lambda_cls=lambda_cls,
        )

    def forward(
        self,
        expr: torch.Tensor,        # (B, G)
        pert_mask: torch.Tensor,   # (B, G) binary
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        반환:
            pred_delta_expr:  (B, G) 예측 발현 변화량
            go_logits:        (B, K) GO 분류 로짓
        """
        x = torch.cat([expr, pert_mask.float()], dim=-1)   # (B, 2G)
        h = self.encoder(x)
        delta_expr = self.expr_head(h)
        go_logits = self.go_head(h)
        return delta_expr, go_logits

    def compute_loss(
        self,
        expr: torch.Tensor,
        pert_mask: torch.Tensor,
        true_delta_expr: torch.Tensor,   # (B, G) 실제 발현 변화량
        go_labels: torch.Tensor,         # (B, K) GO multi-label
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_delta, go_logits = self(expr, pert_mask)
        return self.criterion(pred_delta, true_delta_expr, go_logits, go_labels)

    @torch.no_grad()
    def predict_with_hierarchy(
        self,
        expr: torch.Tensor,
        pert_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        예측 결과 + 계층 구조 시각화용 데이터 반환.
        """
        pred_delta, go_logits = self(expr, pert_mask)
        go_probs = torch.sigmoid(go_logits)
        # 조상 확률도 전파
        anc_probs = torch.matmul(go_probs, self.criterion.ancestor_matrix).clamp(0, 1)
        return {
            "pred_delta_expr": pred_delta,
            "go_probs": go_probs,
            "all_node_probs": anc_probs,
        }
