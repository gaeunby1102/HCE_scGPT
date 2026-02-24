"""
scgpt_hce.py
------------
scGPT 파인튜닝에 HCE 손실 통합.

scGPT 학습 루프(trainer.py)를 직접 수정하지 않고,
HCE 헤드가 부착된 래퍼 모델과 HCE-aware train 함수를 제공.

핵심 아이디어:
    scGPT forward() → cell embedding → HCE GO 헤드 추가
    train 루프에서 기존 loss + λ·HCE_loss

실행:
    python -m HCE.scgpt_hce
"""

from __future__ import annotations
import sys
import os
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# scGPT + HCE 래퍼 모델
# ======================================================================

class ScGPTWithHCE(nn.Module):
    """
    scGPT 모델에 GO 분류 헤드를 추가한 래퍼.

    scGPT의 cell embedding (CLS 토큰 또는 평균 풀링)을 가로채서
    GO term 분류 로짓을 추가로 생성.

    변경점:
    - __init__: go_head (MLP) 추가
    - forward: (original_output_dict, go_logits) 반환
    """

    def __init__(self, scgpt_model: nn.Module, n_go: int, d_model: int = 512):
        super().__init__()
        self.scgpt = scgpt_model
        self.n_go = n_go

        # GO 분류 헤드: cell embedding → GO logits
        self.go_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_go),
        )

    def forward(self, *args, **kwargs) -> Tuple[Dict, torch.Tensor]:
        """
        scGPT forward 그대로 실행 후 cell_emb에서 GO 로짓 계산.

        scGPT output_dict 키:
            - 'mlm_output': MLM 예측값
            - 'cell_emb':   CLS 토큰 임베딩 (CLS=True일 때)
            - 'cls_output': 세포 유형 분류 로짓 (CLS=True일 때)
        """
        output_dict = self.scgpt(*args, **kwargs)

        # cell_emb 추출 (CLS 토큰 or 평균)
        if "cell_emb" in output_dict:
            cell_emb = output_dict["cell_emb"]          # (B, d_model)
        else:
            # mlm_output: (B, seq_len, d_model) → 평균
            cell_emb = output_dict.get("mlm_output", None)
            if cell_emb is not None and cell_emb.dim() == 3:
                cell_emb = cell_emb.mean(dim=1)
            else:
                # fallback: zero
                cell_emb = torch.zeros(
                    args[0].shape[0], self.go_head[0].in_features,
                    device=args[0].device
                )

        go_logits = self.go_head(cell_emb)  # (B, n_go)
        return output_dict, go_logits


# ======================================================================
# HCE-aware 학습 함수 (scGPT trainer.train 대체)
# ======================================================================

def train_one_epoch_hce(
    model: ScGPTWithHCE,
    loader,
    vocab,
    criterion_gep_gepc,
    criterion_dab,
    criterion_cls,
    hce_criterion,          # HierarchicalPerturbationLoss
    go_label_fn,            # batch_data → go_labels tensor 생성 함수
    scaler,
    optimizer,
    scheduler,
    device,
    config,
    lambda_hce: float = 0.3,
    epoch: int = 0,
):
    """
    scGPT trainer.train()에 HCE 손실을 추가한 드롭인 대체 함수.

    기존 scGPT train()과 시그니처가 거의 동일.
    추가 파라미터: hce_criterion, go_label_fn, lambda_hce

    go_label_fn:
        batch_data → torch.Tensor (B, n_go) {0,1} 라벨 생성 함수.
        예: lambda bd: make_go_labels_from_cell_types(bd['celltype_labels'], ...)
    """
    from scgpt.loss import criterion_neg_log_bernoulli

    model.train()
    total_loss = total_gep = total_cls = total_hce = 0.0
    n_batches = len(loader)

    for batch_i, batch_data in enumerate(loader):
        input_gene_ids  = batch_data["gene_ids"].to(device)
        input_values    = batch_data["values"].to(device)
        target_values   = batch_data["target_values"].to(device)
        batch_labels    = batch_data["batch_labels"].to(device)

        if config.task == "annotation":
            celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])

        with torch.cuda.amp.autocast(enabled=config.amp):
            # ① scGPT forward + HCE 헤드
            output_dict, go_logits = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config.use_batch_labels else None,
                CLS=config.CLS,
                MVC=config.GEPC,
                ECS=config.ESC,
            )

            masked_positions = input_values.eq(config.mask_value)
            loss = torch.tensor(0.0, device=device)

            # ② 기존 scGPT 손실들
            if config.GEP:
                loss_gep = criterion_gep_gepc(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_gep
                total_gep += loss_gep.item()

            if config.CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                total_cls += loss_cls.item()

            if config.DAR:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + config.dab_weight * loss_dab

            # ③ HCE 손실 (GO 계층 분류)
            go_labels = go_label_fn(batch_data).to(device)
            loss_hce = hce_criterion._hierarchical_multilabel_loss(go_logits, go_labels)
            loss = loss + lambda_hce * loss_hce
            total_hce += loss_hce.item()

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    scheduler.step()
    n = max(n_batches, 1)
    print(f"  [scGPT+HCE] Epoch {epoch+1} | "
          f"total={total_loss/n:.4f} "
          f"gep={total_gep/n:.4f} "
          f"cls={total_cls/n:.4f} "
          f"hce={total_hce/n:.4f}")


# ======================================================================
# 빠른 통합 테스트 (scGPT 모델 없이 구조만 검증)
# ======================================================================

def _mock_scgpt_forward(gene_ids, values, **kwargs):
    B = gene_ids.shape[0]
    d = 512
    return {
        "mlm_output": torch.randn(B, gene_ids.shape[1], d, device=gene_ids.device),
        "cell_emb":   torch.randn(B, d, device=gene_ids.device),
        "cls_output": torch.randn(B, 10, device=gene_ids.device),
    }


class MockScGPT(nn.Module):
    def forward(self, gene_ids, values, **kwargs):
        return _mock_scgpt_forward(gene_ids, values, **kwargs)


def test_scgpt_hce_structure():
    """scGPT + HCE 래퍼 구조 검증 (실제 scGPT 모델 없이)."""
    from HCE.msigdb_ontology import build_hallmark_ontology
    from HCE.loss import HierarchicalPerturbationLoss

    dag, term_to_idx, _ = build_hallmark_ontology()
    n_go = len(term_to_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mock_scgpt = MockScGPT().to(device)
    model = ScGPTWithHCE(mock_scgpt, n_go=n_go, d_model=512).to(device)

    hce_criterion = HierarchicalPerturbationLoss(
        ontology=dag,
        go_term_to_idx=term_to_idx,
        lambda_reg=1.0,
        lambda_cls=0.3,
    ).to(device)

    # 더미 배치
    B, seq_len = 8, 200
    gene_ids = torch.randint(0, 1000, (B, seq_len)).to(device)
    values   = torch.randn(B, seq_len).to(device)

    output_dict, go_logits = model(gene_ids, values)
    go_labels = (torch.rand(B, n_go) > 0.7).float().to(device)
    hce_loss  = hce_criterion._hierarchical_multilabel_loss(go_logits, go_labels)

    print(f"[scGPT+HCE 구조 검증]")
    print(f"  cell_emb shape: {output_dict['cell_emb'].shape}")
    print(f"  go_logits shape: {go_logits.shape}")
    print(f"  HCE loss: {hce_loss.item():.4f}")
    print(f"  n_go={n_go}, DAG={dag}")
    print("  ✓ scGPT + HCE 통합 구조 정상")


if __name__ == "__main__":
    test_scgpt_hce_structure()
