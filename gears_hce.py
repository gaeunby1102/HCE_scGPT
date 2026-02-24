"""
gears_hce.py
------------
GEARS_Model + HCE 헤드를 서브클래싱으로 통합.
gears2 conda 환경에서 실행해야 함.

사용 예시:
    /home/t1/miniconda3/envs/gears2/bin/python -m HCE.gears_hce

핵심 변경사항
-------------
GEARSModelWithHCE(GEARS_Model):
    - 기존 GEARS_Model의 forward() 출력 위에 GO head 추가
    - hidden 표현(base_emb)을 공유해서 GO 분류 수행

GEARSWithHCE(GEARS):
    - train() 루프에서 기존 loss_fct에 HCE loss를 추가
    - GO 라벨: 배치에서 섭동된 유전자 → 경로 딕셔너리로 실시간 생성
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, "/data2/Atlas_Normal")

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# GEARS (gears2 env)
from gears.model import GEARS_Model, MLP
from gears.gears import GEARS
from gears.utils import loss_fct, print_sys
from gears.inference import evaluate, compute_metrics

# HCE 모듈
from HCE.data_replogle import (
    build_k562_go_ontology,
    PATHWAY_GENES,
    PATHWAY_TO_IDX,
    N_PATHWAYS,
    LEAF_PATHWAYS,
)
from HCE.loss import HierarchicalPerturbationLoss


# ======================================================================
# GO 라벨 실시간 생성 (배치용)
# ======================================================================

def make_go_labels_from_perts(
    perts: List[str],
    gene_list: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    배치의 섭동 유전자명으로 GO multi-label 행렬을 생성.

    Args
    ----
    perts:     배치의 섭동 조건 문자열 (예: "BRCA1+ctrl")
    gene_list: 전체 유전자 목록
    device:    텐서 디바이스

    Returns
    -------
    go_labels: (B, N_PATHWAYS) {0,1} float tensor
    """
    B = len(perts)
    go_labels = torch.zeros(B, N_PATHWAYS, device=device)

    for i, pert_str in enumerate(perts):
        # GEARS 조건 형식: "GENE+ctrl" 또는 "ctrl"
        pert_genes = [p for p in pert_str.split("+") if p != "ctrl"]
        for gene in pert_genes:
            for pathway, idx in PATHWAY_TO_IDX.items():
                if gene in PATHWAY_GENES[pathway]:
                    go_labels[i, idx] = 1.0

    return go_labels


# ======================================================================
# HCE 헤드가 추가된 GEARS 모델
# ======================================================================

class GEARSModelWithHCE(GEARS_Model):
    """
    GEARS_Model + GO 분류 헤드.

    기존 GEARS_Model의 forward()에서 hidden 표현을 중간에 가로채서
    GO 분류 로짓을 추가로 출력.

    변경점
    ------
    - __init__: go_head (MLP) 추가
    - forward: (pred_expr, go_logits) 튜플 반환
    """

    def __init__(self, args: dict, n_go: int):
        super().__init__(args)
        hidden_size = args["hidden_size"]
        self.n_go = n_go

        # GO 분류 헤드: latent → GO logits
        self.go_head = nn.Sequential(
            nn.Linear(hidden_size * args["num_genes"], hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, n_go),
        )

        # 실제로는 base_emb가 (B*G, hidden) 형태라 pooling이 필요
        # 더 가벼운 대안: gene별 평균 풀링 후 분류
        self.go_pool = nn.AdaptiveAvgPool1d(1)
        self.go_head_light = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, n_go),
        )

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pred:       (B, G) 예측 발현량 (기존 GEARS 출력)
        go_logits:  (B, N_GO) GO 분류 로짓 (HCE용)
        """
        x, pert_idx = data.x, data.pert_idx

        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            pred = torch.stack(out)
            go_logits = torch.zeros(pred.shape[0], self.n_go, device=x.device)
            return pred, go_logits

        num_graphs = len(data.batch.unique())

        # ── 기존 GEARS forward 재현 (base_emb 중간 추출) ──────────
        emb = self.gene_emb(
            torch.LongTensor(list(range(self.num_genes)))
            .repeat(num_graphs).to(self.args["device"])
        )
        emb = self.bn_emb(emb)
        base_emb = self.emb_trans(emb)

        pos_emb = self.emb_pos(
            torch.LongTensor(list(range(self.num_genes)))
            .repeat(num_graphs).to(self.args["device"])
        )
        for idx, layer in enumerate(self.layers_emb_pos):
            pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
            if idx < len(self.layers_emb_pos) - 1:
                pos_emb = pos_emb.relu()

        base_emb = base_emb + 0.2 * pos_emb
        base_emb = self.emb_trans_v2(base_emb)  # (B*G, H)

        pert_index = []
        for idx, i in enumerate(pert_idx):
            for j in i:
                if j != -1:
                    pert_index.append([idx, j])
        pert_index = torch.tensor(pert_index).T

        pert_global_emb = self.pert_emb(
            torch.LongTensor(list(range(self.num_perts))).to(self.args["device"])
        )
        for idx, layer in enumerate(self.sim_layers):
            pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_global_emb = pert_global_emb.relu()

        base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

        if pert_index.shape[0] != 0:
            pert_track = {}
            for i, j in enumerate(pert_index[0]):
                if j.item() in pert_track:
                    pert_track[j.item()] += pert_global_emb[pert_index[1][i]]
                else:
                    pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

            if len(list(pert_track.values())) > 0:
                if len(list(pert_track.values())) == 1:
                    emb_total = self.pert_fuse(
                        torch.stack(list(pert_track.values()) * 2)
                    )
                else:
                    emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                for idx, j in enumerate(pert_track.keys()):
                    base_emb[j] = base_emb[j] + emb_total[idx]

        # ── GO 헤드: gene 축으로 평균 풀링 ───────────────────────
        # base_emb: (B, G, H) → mean over G → (B, H)
        graph_emb = base_emb.mean(dim=1)              # (B, H)
        go_logits = self.go_head_light(graph_emb)     # (B, N_GO)

        # ── 기존 GEARS decoder 계속 ───────────────────────────────
        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        base_emb = self.bn_pert_base(base_emb)
        base_emb = self.transform(base_emb)
        out = self.recovery_w(base_emb)
        out = out.reshape(num_graphs, self.num_genes, -1)
        out = out.unsqueeze(-1) * self.indv_w1
        w = torch.sum(out, axis=2)
        out = w + self.indv_b1

        cross_gene_embed = self.cross_gene_state(
            out.reshape(num_graphs, self.num_genes, -1).squeeze(2)
        )
        cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)
        cross_gene_embed = cross_gene_embed.reshape([num_graphs, self.num_genes, -1])
        cross_gene_out = torch.cat([out, cross_gene_embed], 2)
        cross_gene_out = cross_gene_out * self.indv_w2
        cross_gene_out = torch.sum(cross_gene_out, axis=2)
        out = cross_gene_out + self.indv_b2
        out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
        out = torch.split(torch.flatten(out), self.num_genes)
        pred = torch.stack(out)

        return pred, go_logits


# ======================================================================
# HCE 통합 GEARS 트레이너
# ======================================================================

class GEARSWithHCE(GEARS):
    """
    GEARS 트레이너 + HCE 손실 통합.

    train() 루프를 오버라이드하여:
    1. 기존 MSE + direction loss (loss_fct) 유지
    2. GO 분류 HCE loss 추가

    model_initialize_hce()로 초기화 (기존 model_initialize 대체).
    """

    def model_initialize_hce(
        self,
        hidden_size: int = 64,
        num_go_gnn_layers: int = 1,
        num_gene_gnn_layers: int = 1,
        decoder_hidden_size: int = 16,
        num_similar_genes_go_graph: int = 20,
        num_similar_genes_co_express_graph: int = 20,
        coexpress_threshold: float = 0.4,
        direction_lambda: float = 1e-1,
        lambda_hce: float = 0.3,
        G_go=None,
        G_go_weight=None,
        G_coexpress=None,
        G_coexpress_weight=None,
    ):
        """
        HCE 통합 모델 초기화.

        Args
        ----
        lambda_hce: HCE loss 가중치 (기존 loss에 더하는 비율)
        나머지 인자: 기존 GEARS model_initialize와 동일
        """
        self.lambda_hce = lambda_hce

        # GO 온톨로지 구성
        self.hce_dag, self.hce_term_to_idx = build_k562_go_ontology()
        n_go = len(self.hce_term_to_idx)

        # 기존 GEARS config 구성 (model_initialize 로직 재사용)
        self.model_initialize(
            hidden_size=hidden_size,
            num_go_gnn_layers=num_go_gnn_layers,
            num_gene_gnn_layers=num_gene_gnn_layers,
            decoder_hidden_size=decoder_hidden_size,
            num_similar_genes_go_graph=num_similar_genes_go_graph,
            num_similar_genes_co_express_graph=num_similar_genes_co_express_graph,
            coexpress_threshold=coexpress_threshold,
            uncertainty=False,
            direction_lambda=direction_lambda,
            G_go=G_go,
            G_go_weight=G_go_weight,
            G_coexpress=G_coexpress,
            G_coexpress_weight=G_coexpress_weight,
        )

        # GEARS_Model → GEARSModelWithHCE 교체
        self.model = GEARSModelWithHCE(self.config, n_go=n_go).to(self.device)
        self.best_model = deepcopy(self.model)

        # HCE 손실함수
        self.hce_criterion = HierarchicalPerturbationLoss(
            ontology=self.hce_dag,
            go_term_to_idx=self.hce_term_to_idx,
            lambda_reg=1.0,
            lambda_cls=lambda_hce,
        ).to(self.device)

        print_sys(
            f"[HCE] GO terms: {n_go}, lambda_hce: {lambda_hce}, "
            f"n_genes: {self.num_genes}, n_perts: {self.num_perts}"
        )

    def train(
        self,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
    ):
        """
        HCE 통합 학습 루프.

        기존 loss_fct + HCE GO 분류 loss를 결합.
        """
        train_loader = self.dataloader["train_loader"]
        val_loader   = self.dataloader["val_loader"]

        self.model = self.model.to(self.device)
        best_model  = deepcopy(self.model)
        optimizer   = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler   = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print_sys("Start Training with HCE...")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = epoch_mse = epoch_hce = 0.0

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()

                # ① 모델 forward (pred_expr, go_logits)
                pred, go_logits = self.model(batch)
                y = batch.y

                # ② 기존 MSE + direction loss
                mse_loss = loss_fct(
                    pred, y, batch.pert,
                    ctrl=self.ctrl_expression,
                    dict_filter=self.dict_filter,
                    direction_lambda=self.config["direction_lambda"],
                )

                # ③ GO HCE loss (배치에서 GO 라벨 실시간 생성)
                go_labels = make_go_labels_from_perts(
                    list(batch.pert), self.gene_list, self.device
                )
                # HierarchicalPerturbationLoss.forward 시그니처:
                # (pred_expr, true_expr, go_logits, go_labels)
                # 여기서는 go_logits만 사용하는 subset 계산
                hce_loss = self.hce_criterion._hierarchical_multilabel_loss(
                    go_logits, go_labels
                )

                # ④ 결합
                total_loss = mse_loss + self.lambda_hce * hce_loss
                total_loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_mse  += mse_loss.item()
                epoch_hce  += hce_loss.item()

                if step % 50 == 0:
                    print_sys(
                        f"Epoch {epoch+1} Step {step+1} | "
                        f"total={total_loss.item():.4f} "
                        f"mse={mse_loss.item():.4f} "
                        f"hce={hce_loss.item():.4f}"
                    )

            scheduler.step()

            # 검증 (기존 evaluate는 pred만 반환받도록 monkey-patch 아닌 래핑)
            val_res  = self._evaluate_hce(val_loader)
            val_mse  = np.mean([v["mse"] for v in val_res.values()])
            val_pear = np.mean([v["pearson"] for v in val_res.values()])

            print_sys(
                f"Epoch {epoch+1}: "
                f"Val MSE={val_mse:.4f} | Val Pearson={val_pear:.4f} | "
                f"HCE loss={epoch_hce/max(step+1,1):.4f}"
            )

            if val_mse < min_val:
                min_val = val_mse
                best_model = deepcopy(self.model)

        self.best_model = best_model
        print_sys("Done!")

    def _evaluate_hce(self, loader) -> Dict:
        """
        HCE 모델용 간단한 검증 (MSE + Pearson per perturbation).
        """
        self.model.eval()
        results = {}
        with torch.no_grad():
            for batch in loader:
                batch.to(self.device)
                pred, _ = self.model(batch)  # go_logits 무시
                y = batch.y
                perts = np.array(batch.pert)
                for p in set(perts):
                    idx = np.where(perts == p)[0]
                    p_pred = pred[idx].cpu().numpy()
                    p_true = y[idx].cpu().numpy()
                    mse = ((p_pred - p_true) ** 2).mean()
                    # Pearson (전체 유전자)
                    from scipy.stats import pearsonr
                    r_vals = [pearsonr(p_pred[i], p_true[i])[0] for i in range(len(idx))]
                    if p not in results:
                        results[p] = {"mse": [], "pearson": []}
                    results[p]["mse"].append(mse)
                    results[p]["pearson"].append(np.mean(r_vals))
        # 평균화
        return {
            p: {"mse": np.mean(v["mse"]), "pearson": np.mean(v["pearson"])}
            for p, v in results.items()
        }


# ======================================================================
# 직접 실행 시 빠른 통합 테스트 (PertData 없이 구조만 검증)
# ======================================================================

if __name__ == "__main__":
    print("GEARSWithHCE 구조 검증 (PertData 없이 모듈 import 테스트)")
    dag, term_to_idx = build_k562_go_ontology()
    print(f"GO 온톨로지: {dag}")
    print(f"리프 경로 ({len(term_to_idx)}개):")
    for t, i in term_to_idx.items():
        print(f"  [{i}] {t}")

    # GEARSModelWithHCE 단독 초기화 테스트 (그래프 없이)
    print("\nGEARSModelWithHCE 단독 import 성공.")
    print("실제 학습은 PertData 로딩 후 GEARSWithHCE.train()으로 수행.")
