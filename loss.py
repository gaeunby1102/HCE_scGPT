"""
loss.py
-------
HierarchicalCrossEntropyLoss (HCE)
- Nature Computational Science (2025) 아이디어를 기반으로 구현
- 세포 유형 분류 + 섭동 결과 분류 두 모드 지원
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ontology import OntologyDAG


class HierarchicalCrossEntropyLoss(nn.Module):
    """
    Hierarchical Cross-Entropy Loss.

    핵심 아이디어
    -------------
    표준 CE:  L = CE(logits, leaf_label)
    HCE:      L = Σ_k  w_k · CE(P_k, y_k)
        - k: 온톨로지의 각 레벨
        - P_k: k 레벨에서의 집계 확률 (리프 확률을 상위로 합산)
        - y_k: k 레벨에서의 정답 (리프 라벨의 조상 중 해당 레벨 노드)
        - w_k: 레벨별 가중치 (상위 레벨일수록 더 높은 가중치)

    이 손실함수를 최소화하면:
    1. 리프 수준 분류가 정확해지고 (flat CE)
    2. 틀려도 상위 분류는 맞는 방향으로 학습 (hierarchical robustness → OOD 강건성)
    3. 예측 확률이 부모 >= 자식 조건을 만족 (monotonicity)

    Args
    ----
    ontology:      OntologyDAG 객체
    term_to_idx:   분류 대상 리프 노드 → 인덱스 매핑
    alpha:         flat CE vs hierarchical CE 비율 (0 = flat only, 1 = hier only)
    monotone_coef: 단조성 패널티 계수
    level_decay:   레벨 가중치 지수 감쇠율 (상위 레벨에 더 큰 가중치)
    """

    def __init__(
        self,
        ontology: OntologyDAG,
        term_to_idx: Dict[str, int],
        alpha: float = 0.7,
        monotone_coef: float = 0.1,
        level_decay: float = 0.5,
    ):
        super().__init__()
        self.ontology = ontology
        self.term_to_idx = term_to_idx
        self.idx_to_term = {v: k for k, v in term_to_idx.items()}
        self.n_classes = len(term_to_idx)
        self.alpha = alpha
        self.monotone_coef = monotone_coef
        self.level_decay = level_decay

        # 각 리프 노드의 조상 집합 미리 계산
        self._ancestors: Dict[int, List[int]] = {}
        self._build_ancestor_index()

        # 레벨별 노드 그룹 구성 (계층적 CE 계산용)
        self._level_groups = self._build_level_groups()

        # ancestor matrix (leaf_idx -> all_term_idx)
        # [i, j] = 1 이면 j번째 전체 노드가 i번째 리프의 조상
        all_terms = list(ontology.nodes.keys())
        self.all_term_to_idx = {t: i for i, t in enumerate(all_terms)}

        anc_matrix = torch.zeros(self.n_classes, len(all_terms))
        for leaf_term, leaf_idx in term_to_idx.items():
            for anc in ontology.get_ancestors(leaf_term, include_self=True):
                if anc in self.all_term_to_idx:
                    anc_matrix[leaf_idx, self.all_term_to_idx[anc]] = 1.0
        self.register_buffer("ancestor_matrix", anc_matrix)  # (n_classes, n_all_terms)

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _build_ancestor_index(self) -> None:
        """리프 인덱스 → 조상 리프 인덱스 매핑 (같은 리프 집합 내)."""
        for term, idx in self.term_to_idx.items():
            ancestors = self.ontology.get_ancestors(term, include_self=True)
            # 조상 중 term_to_idx에 있는 것만 (리프끼리 상호 조상 관계는 없지만 혹시 모르므로)
            self._ancestors[idx] = [
                self.term_to_idx[a] for a in ancestors if a in self.term_to_idx
            ]

    def _build_level_groups(self) -> List[List[str]]:
        """
        온톨로지를 레벨별로 그룹화.
        반환: [ [depth_0_terms], [depth_1_terms], ... ]
        """
        depth_map: Dict[int, List[str]] = {}
        for term in self.ontology.nodes:
            d = self.ontology.get_depth(term)
            depth_map.setdefault(d, []).append(term)
        max_depth = max(depth_map.keys()) if depth_map else 0
        return [depth_map.get(d, []) for d in range(max_depth + 1)]

    # ------------------------------------------------------------------
    # 확률 전파 (리프 → 상위 노드)
    # ------------------------------------------------------------------

    def propagate_probs(self, leaf_probs: torch.Tensor) -> torch.Tensor:
        """
        리프 수준 확률을 상위 노드로 전파 (합산).
        leaf_probs: (B, n_classes)  softmax 이후 확률
        반환:       (B, n_all_terms) 전체 노드의 확률
        """
        # all_term_probs[b, j] = Σ_{i: leaf i의 조상에 j 포함} leaf_probs[b, i]
        all_term_probs = torch.matmul(leaf_probs, self.ancestor_matrix)
        # 클리핑: 여러 자식의 합이 1을 넘을 수 있음 (특히 루트 근처)
        all_term_probs = all_term_probs.clamp(0.0, 1.0)
        return all_term_probs

    # ------------------------------------------------------------------
    # 라벨 전파 (리프 라벨 → 상위 이진 라벨)
    # ------------------------------------------------------------------

    def propagate_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        리프 라벨을 조상 노드로 전파하여 이진 행렬 생성.
        labels: (B,) 정수 인덱스
        반환:  (B, n_all_terms) {0, 1} 행렬
        """
        B = labels.shape[0]
        one_hot = torch.zeros(B, self.n_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)           # (B, n_classes)
        binary = torch.matmul(one_hot, self.ancestor_matrix)    # (B, n_all_terms)
        return (binary > 0).float()

    # ------------------------------------------------------------------
    # 손실 계산
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args
        ----
        logits: (B, n_classes)  — 모델 출력 (softmax 전)
        labels: (B,)            — 리프 클래스 인덱스

        Returns
        -------
        loss:  스칼라 손실값
        info:  각 구성 요소 딕셔너리 (로깅용)
        """
        # ① 표준 평탄 CE (리프 수준)
        flat_ce = F.cross_entropy(logits, labels)

        # ② 계층적 BCE (모든 조상 노드에서 이진 분류)
        leaf_probs = F.softmax(logits, dim=-1)           # (B, n_classes)
        all_probs = self.propagate_probs(leaf_probs)     # (B, n_all_terms)
        hier_labels = self.propagate_labels(labels)      # (B, n_all_terms)

        # 각 노드의 깊이 기반 가중치 (얕을수록 더 중요 — 상위 분류 먼저 맞추기)
        depths = torch.tensor(
            [self.ontology.get_depth(t) for t in self.ontology.nodes.keys()],
            dtype=torch.float32,
            device=logits.device,
        )
        max_depth = depths.max().clamp(min=1)
        # 깊이 0 (루트) → weight 1.0, 깊이 max → weight level_decay^max
        level_weights = self.level_decay ** depths          # (n_all_terms,)
        level_weights = level_weights / level_weights.sum()

        # BCE per node per sample
        eps = 1e-7
        all_probs_clipped = all_probs.clamp(eps, 1 - eps)
        bce_per_node = -(
            hier_labels * all_probs_clipped.log()
            + (1 - hier_labels) * (1 - all_probs_clipped).log()
        )  # (B, n_all_terms)

        hier_ce = (bce_per_node * level_weights.unsqueeze(0)).sum(dim=1).mean()

        # ③ 단조성 패널티: P(자식) ≤ P(부모) 강제
        # 각 리프에 대해 조상 확률이 자신보다 크거나 같아야 함
        mono_loss = self._monotonicity_penalty(leaf_probs, all_probs)

        # ④ 결합
        total = (1 - self.alpha) * flat_ce + self.alpha * hier_ce + self.monotone_coef * mono_loss

        info = {
            "loss_total":      total.item(),
            "loss_flat_ce":    flat_ce.item(),
            "loss_hier_ce":    hier_ce.item(),
            "loss_monotone":   mono_loss.item(),
        }
        return total, info

    def _monotonicity_penalty(
        self,
        leaf_probs: torch.Tensor,   # (B, n_classes)
        all_probs: torch.Tensor,    # (B, n_all_terms)
    ) -> torch.Tensor:
        """
        각 리프 확률이 자신의 조상 확률보다 크지 않도록 패널티.
        penalty = mean(ReLU(P_leaf - min_ancestor_prob))
        """
        penalties = []
        for leaf_term, leaf_idx in self.term_to_idx.items():
            ancestors = self.ontology.get_ancestors(leaf_term, include_self=False)
            if not ancestors:
                continue
            p_leaf = leaf_probs[:, leaf_idx]  # (B,)
            anc_indices = [
                self.all_term_to_idx[a]
                for a in ancestors
                if a in self.all_term_to_idx
            ]
            if not anc_indices:
                continue
            p_anc = all_probs[:, anc_indices]           # (B, n_anc)
            p_anc_min = p_anc.min(dim=1).values         # (B,)
            penalty = F.relu(p_leaf - p_anc_min)        # (B,)
            penalties.append(penalty)

        if not penalties:
            return torch.tensor(0.0, device=leaf_probs.device)
        return torch.stack(penalties, dim=1).mean()


# ======================================================================
# 섭동 모델 전용: Hierarchical Multi-Label Loss (GO 기반)
# ======================================================================

class HierarchicalPerturbationLoss(nn.Module):
    """
    유전자 섭동 결과를 GO 계층 구조로 분류하기 위한 손실함수.

    회귀(연속 발현량 예측) + 계층 분류(어떤 GO 프로세스가 영향받았나)를
    결합한 복합 손실함수.

    L_total = λ_reg · L_regression + λ_cls · L_HCE_multilabel

    섭동 결과는 단일 라벨이 아니라 여러 GO 텀이 동시에 활성화될 수 있으므로
    multi-label BCE를 계층적으로 적용.
    """

    def __init__(
        self,
        ontology: OntologyDAG,
        go_term_to_idx: Dict[str, int],
        lambda_reg: float = 1.0,
        lambda_cls: float = 0.5,
        level_decay: float = 0.5,
    ):
        super().__init__()
        self.ontology = ontology
        self.go_term_to_idx = go_term_to_idx
        self.idx_to_term = {v: k for k, v in go_term_to_idx.items()}
        self.n_go = len(go_term_to_idx)
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.level_decay = level_decay

        # ancestor matrix (n_go, n_all_terms)
        all_terms = list(ontology.nodes.keys())
        self.all_term_to_idx = {t: i for i, t in enumerate(all_terms)}

        anc_matrix = torch.zeros(self.n_go, len(all_terms))
        for term, idx in go_term_to_idx.items():
            for anc in ontology.get_ancestors(term, include_self=True):
                if anc in self.all_term_to_idx:
                    anc_matrix[idx, self.all_term_to_idx[anc]] = 1.0
        self.register_buffer("ancestor_matrix", anc_matrix)

        # 레벨 가중치
        depths = torch.tensor(
            [ontology.get_depth(t) for t in all_terms],
            dtype=torch.float32,
        )
        weights = level_decay ** depths
        self.register_buffer("level_weights", weights / weights.sum())

    def forward(
        self,
        pred_expr: torch.Tensor,          # (B, n_genes) 예측 발현량
        true_expr: torch.Tensor,          # (B, n_genes) 실제 발현량
        go_logits: torch.Tensor,          # (B, n_go) GO 분류 로짓
        go_labels: torch.Tensor,          # (B, n_go) {0,1} multi-label
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args
        ----
        pred_expr:  예측 유전자 발현량
        true_expr:  실측 발현량
        go_logits:  GO 텀 로짓 (sigmoid 전)
        go_labels:  GO 텀 이진 라벨 (어떤 프로세스가 활성화됐나)

        Returns
        -------
        loss, info
        """
        # ① 회귀 손실 (MSE)
        reg_loss = F.mse_loss(pred_expr, true_expr)

        # ② 계층적 multi-label 분류 손실
        hier_cls = self._hierarchical_multilabel_loss(go_logits, go_labels)

        total = self.lambda_reg * reg_loss + self.lambda_cls * hier_cls
        info = {
            "loss_total":      total.item(),
            "loss_regression": reg_loss.item(),
            "loss_hier_cls":   hier_cls.item(),
        }
        return total, info

    def _hierarchical_multilabel_loss(
        self,
        go_logits: torch.Tensor,   # (B, n_go)
        go_labels: torch.Tensor,   # (B, n_go)
    ) -> torch.Tensor:
        """
        GO 텀 multi-label 분류에 계층적 BCE 적용.
        활성화된 GO 텀의 모든 조상도 활성화로 간주.
        """
        B = go_logits.shape[0]
        go_probs = torch.sigmoid(go_logits)  # (B, n_go)

        # 라벨 전파: 활성화된 GO 텀의 조상도 1로
        # go_labels: (B, n_go) → 조상 행렬 적용 → (B, n_all_terms)
        hier_labels = torch.matmul(go_labels.float(), self.ancestor_matrix)
        hier_labels = (hier_labels > 0).float()

        # 확률 전파: 합산
        all_probs = torch.matmul(go_probs, self.ancestor_matrix).clamp(0, 1)

        eps = 1e-7
        all_probs = all_probs.clamp(eps, 1 - eps)
        bce = -(
            hier_labels * all_probs.log()
            + (1 - hier_labels) * (1 - all_probs).log()
        )  # (B, n_all_terms)

        weighted_bce = (bce * self.level_weights.unsqueeze(0)).sum(dim=1).mean()
        return weighted_bce
