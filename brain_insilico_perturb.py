"""
brain_insilico_perturb.py
--------------------------
scGPT_brain + HCE 분류 헤드를 이용한 뇌 세포 in silico 유전자 섭동.

실험 설계:
  1. RG (Radial Glia) 세포를 로드
  2. 각 유전자 KO를 시뮬레이션 (발현값 → 0, 즉 패드 값으로 마스킹)
  3. ΔP(Neuroblast) = P(Neuroblast | KO) - P(Neuroblast | 정상) 계산
  4. RG → Neuroblast 전이를 유도하는 상위 유전자 순위화

실행:
    conda run -n scgpt python -m HCE.brain_insilico_perturb

출력:
    HCE/results/brain_insilico_perturb.json
    HCE/results/brain_insilico_perturb.log
"""

from __future__ import annotations
import sys, os, json, warnings, time
sys.path.insert(0, "/data2/Atlas_Normal")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from torch.utils.data import Dataset, DataLoader

SCGPT_DIR  = "/data4/scGPT_brain"
DATA_PATH  = "/data2/Atlas_Normal/IL17RD_scdiffeq/results/Input_fetal_neuron_subset_harmony_integration_250106.h5ad"
MODEL_PATH = "/data2/Atlas_Normal/HCE/jacobian/results/hce_brain_best.pt"
RESULTS_DIR = "/data2/Atlas_Normal/HCE/results"
LOG_PATH   = os.path.join(RESULTS_DIR, "brain_insilico_perturb.log")
JSON_PATH  = os.path.join(RESULTS_DIR, "brain_insilico_perturb.json")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_RG_CELLS = 200    # RG 세포 샘플 수 (더 많으면 정확하지만 느림)
MAX_SEQ    = 1200   # scGPT 시퀀스 길이
N_BINS     = 51
N_TOP_GENES = 300   # 테스트할 유전자 수 (RG 세포에서 발현량 상위)
BATCH_SIZE  = 32    # GPU 배치

CELL_TYPES  = ["RG", "Neuroblast", "Ext", "Inh"]
LEAF_TO_IDX = {"RG": 0, "Neuroblast": 1, "Ext": 2, "Inh": 3}
IDX_TO_LEAF = {v: k for k, v in LEAF_TO_IDX.items()}


# ── 로그 ─────────────────────────────────────────────────────────────────────
def log(msg: str):
    print(msg, flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ── Brain Cell Ontology (step1과 동일) ─────────────────────────────────────
def build_brain_cell_ontology():
    from HCE.ontology import OntologyDAG
    dag = OntologyDAG()
    dag.add_edge("Excitatory_Neuron",  "Neuron")
    dag.add_edge("Inhibitory_Neuron",  "Neuron")
    dag.add_edge("Radial_Glia",        "Neural_Progenitor")
    dag.add_edge("Neuroblast_cell",    "Neural_Progenitor")
    dag.add_edge("Neuron",             "Neural_Cell")
    dag.add_edge("Neural_Progenitor",  "Neural_Cell")
    dag.add_edge("Neural_Cell",        "Cell")
    term_to_idx = {
        "Radial_Glia":       LEAF_TO_IDX["RG"],
        "Neuroblast_cell":   LEAF_TO_IDX["Neuroblast"],
        "Excitatory_Neuron": LEAF_TO_IDX["Ext"],
        "Inhibitory_Neuron": LEAF_TO_IDX["Inh"],
    }
    return dag, term_to_idx


# ── ScGPTBrainHCE 모델 (step1과 동일 구조) ────────────────────────────────
class ScGPTBrainHCE(nn.Module):
    def __init__(self, scgpt_model, n_classes, d_model=512):
        super().__init__()
        self.scgpt    = scgpt_model
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, gene_ids, values, src_key_padding_mask):
        output   = self.scgpt(
            gene_ids, values,
            src_key_padding_mask=src_key_padding_mask,
            CLS=False, MVC=False, ECS=False,
        )
        cell_emb = output["cell_emb"]       # (B, d_model)
        logits   = self.cls_head(cell_emb)  # (B, n_classes)
        return logits, cell_emb


# ── scGPT_brain 로드 ──────────────────────────────────────────────────────
def load_scgpt_brain():
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    vocab = GeneVocab.from_file(os.path.join(SCGPT_DIR, "vocab.json"))
    args  = json.load(open(os.path.join(SCGPT_DIR, "args.json")))
    pad_id = vocab[args["pad_token"]]
    model  = TransformerModel(
        ntoken           = len(vocab),
        d_model          = args["embsize"],
        nhead            = args["nheads"],
        d_hid            = args["d_hid"],
        nlayers          = args["nlayers"],
        nlayers_cls      = args.get("n_layers_cls", 3),
        n_cls            = 1,
        vocab            = vocab,
        dropout          = args["dropout"],
        pad_token        = args["pad_token"],
        pad_value        = args["pad_value"],
        do_mvc           = False,
        do_dab           = False,
        use_batch_labels = False,
        input_emb_style  = args["input_emb_style"],
        n_input_bins     = args["n_bins"],
        cell_emb_style   = "cls",
        use_fast_transformer = args.get("fast_transformer", True),
        pre_norm         = False,
    )
    ckpt = torch.load(os.path.join(SCGPT_DIR, "best_model.pt"), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    return model, vocab, args


# ── 세포 → scGPT 입력 변환 ─────────────────────────────────────────────────
def cell_to_scgpt_input(
    expr: np.ndarray,            # (n_genes,) 발현값
    gene_col_idx: list[int],     # adata에서 선택한 gene column 인덱스
    gene_token_ids: list[int],   # vocab 토큰 ID
    max_seq: int,
    n_bins: int,
    pad_id: int,
    pad_value: float = -2.0,
    ko_gene_col: int | None = None,  # 이 gene column을 0으로 만들 (KO)
):
    """단일 세포 발현값 → (gene_ids, values) 텐서 반환."""
    e = expr[gene_col_idx].copy().astype(np.float32)

    if ko_gene_col is not None:
        # ko_gene_col: gene_col_idx 내의 인덱스 (not adata col idx)
        e[ko_gene_col] = 0.0

    n_sel = min(max_seq, len(e))
    top_k = np.argsort(e)[-n_sel:][::-1]

    sel_gene_ids = np.array(gene_token_ids)[top_k]
    sel_vals     = e[top_k]

    sel_vals = np.log1p(sel_vals)
    max_v    = sel_vals.max() + 1e-6
    sel_vals = sel_vals / max_v
    binned   = np.floor(sel_vals * (n_bins - 1)).astype(np.float32)

    pad_len = max_seq - len(sel_gene_ids)
    g_pad   = np.full(pad_len, pad_id, dtype=np.int64)
    v_pad   = np.full(pad_len, pad_value, dtype=np.float32)
    gene_ids = np.concatenate([sel_gene_ids, g_pad])
    vals     = np.concatenate([binned, v_pad])

    return gene_ids, vals


# ── 배치 단위 예측 ─────────────────────────────────────────────────────────
@torch.no_grad()
def predict_batch(
    model, gene_ids_batch, values_batch, pad_id, device
):
    """(B, MAX_SEQ) → (B, 4) softmax 확률."""
    g = torch.tensor(gene_ids_batch, dtype=torch.long,  device=device)
    v = torch.tensor(values_batch,  dtype=torch.float32, device=device)
    mask = g.eq(pad_id)
    logits, _ = model(g, v, mask)
    return F.softmax(logits, dim=-1).cpu().numpy()


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("Brain In Silico Perturbation: scGPT_brain + HCE")
    log(f"목표: RG → Neuroblast 전이 유도 유전자 순위화")
    log("=" * 60)

    # ① scGPT_brain + HCE 모델 로드
    log("\n[1] scGPT_brain + HCE 모델 로드...")
    scgpt_model, vocab, args = load_scgpt_brain()
    for p in scgpt_model.parameters():
        p.requires_grad_(False)

    model = ScGPTBrainHCE(scgpt_model, n_classes=4, d_model=args["embsize"])
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    # step1에서 cls_head만 저장됨: {"epoch": ..., "model_state": cls_head.state_dict(), ...}
    cls_state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.cls_head.load_state_dict(cls_state)
    log(f"  val_acc (step1): {ckpt.get('val_acc', 'N/A')}, epoch: {ckpt.get('epoch', 'N/A')}")
    model = model.to(DEVICE)
    model.eval()

    pad_id    = vocab["<pad>"]
    pad_value = float(args.get("pad_value", -2))
    log(f"  모델 로드 완료: {MODEL_PATH}")
    log(f"  vocab size: {len(vocab)}, d_model: {args['embsize']}")

    # ② 뇌 아틀라스 로드 + RG 세포 샘플링
    log(f"\n[2] 뇌 아틀라스 로드... ({DATA_PATH})")
    adata = sc.read_h5ad(DATA_PATH)
    log(f"  전체 shape: {adata.shape}")

    rg_mask = adata.obs["Cell Type"] == "RG"
    rg_idx  = np.where(rg_mask)[0]
    rng     = np.random.default_rng(42)
    chosen  = rng.choice(rg_idx, min(N_RG_CELLS, len(rg_idx)), replace=False)
    log(f"  RG 세포: {len(rg_idx)}개 중 {len(chosen)}개 샘플링")

    # 공통 유전자 (adata var_names ∩ vocab)
    gene_names = adata.var_names.tolist()
    common = [(i, gene_names[i]) for i in range(len(gene_names)) if gene_names[i] in vocab]
    gene_col_idx   = [c[0] for c in common]
    gene_token_ids = [vocab[c[1]] for c in common]
    log(f"  adata-vocab 겹치는 유전자: {len(common)}/{len(gene_names)}")

    # 발현 행렬 (RG 세포 × 공통 유전자)
    X = adata.layers["logcounts"]
    if hasattr(X, "toarray"): X = X.toarray()
    X = X.astype(np.float32)
    X_rg = X[chosen][:, gene_col_idx]  # (N_RG, n_common)

    # ③ Baseline 예측 (KO 없이)
    log(f"\n[3] Baseline 예측 (N_RG={len(chosen)} 세포)...")
    baseline_probs = []
    gene_ids_list, vals_list = [], []
    for i in range(len(chosen)):
        gids, vals = cell_to_scgpt_input(
            X[chosen[i]], gene_col_idx, gene_token_ids,
            MAX_SEQ, N_BINS, pad_id, pad_value,
            ko_gene_col=None,
        )
        gene_ids_list.append(gids)
        vals_list.append(vals)

    for i in range(0, len(chosen), BATCH_SIZE):
        probs = predict_batch(
            model,
            np.stack(gene_ids_list[i:i+BATCH_SIZE]),
            np.stack(vals_list[i:i+BATCH_SIZE]),
            pad_id, DEVICE,
        )
        baseline_probs.append(probs)
    baseline_probs = np.concatenate(baseline_probs, axis=0)  # (N_RG, 4)

    baseline_rg_prob    = baseline_probs[:, LEAF_TO_IDX["RG"]].mean()
    baseline_nb_prob    = baseline_probs[:, LEAF_TO_IDX["Neuroblast"]].mean()
    baseline_pred_class = baseline_probs.argmax(axis=1)
    baseline_acc        = (baseline_pred_class == LEAF_TO_IDX["RG"]).mean()

    log(f"  Baseline P(RG):          {baseline_rg_prob:.4f}")
    log(f"  Baseline P(Neuroblast):  {baseline_nb_prob:.4f}")
    log(f"  Baseline 정확도 (RG로 예측): {baseline_acc:.4f}")
    log(f"  예측 분포: { {IDX_TO_LEAF[i]: int((baseline_pred_class==i).sum()) for i in range(4)} }")

    # ④ 테스트 유전자 선택: RG 세포에서 발현 상위 유전자
    rg_mean_expr = X_rg.mean(axis=0)  # (n_common,)
    top_gene_positions = np.argsort(rg_mean_expr)[-N_TOP_GENES:][::-1]

    log(f"\n[4] In silico KO 실험: RG 발현 상위 {N_TOP_GENES}개 유전자")
    log(f"  상위 10개 유전자: { [common[i][1] for i in top_gene_positions[:10]] }")

    # ⑤ 각 유전자 KO → ΔP(Neuroblast) 계산
    results_per_gene = {}
    t0 = time.time()

    for rank, pos in enumerate(top_gene_positions):
        gene_name = common[pos][1]

        ko_gene_ids_list, ko_vals_list = [], []
        for i in range(len(chosen)):
            gids, vals = cell_to_scgpt_input(
                X[chosen[i]], gene_col_idx, gene_token_ids,
                MAX_SEQ, N_BINS, pad_id, pad_value,
                ko_gene_col=pos,  # 이 위치 유전자를 0으로
            )
            ko_gene_ids_list.append(gids)
            ko_vals_list.append(vals)

        ko_probs = []
        for i in range(0, len(chosen), BATCH_SIZE):
            probs = predict_batch(
                model,
                np.stack(ko_gene_ids_list[i:i+BATCH_SIZE]),
                np.stack(ko_vals_list[i:i+BATCH_SIZE]),
                pad_id, DEVICE,
            )
            ko_probs.append(probs)
        ko_probs = np.concatenate(ko_probs, axis=0)  # (N_RG, 4)

        delta_nb = (ko_probs[:, LEAF_TO_IDX["Neuroblast"]]
                    - baseline_probs[:, LEAF_TO_IDX["Neuroblast"]]).mean()
        delta_rg = (ko_probs[:, LEAF_TO_IDX["RG"]]
                    - baseline_probs[:, LEAF_TO_IDX["RG"]]).mean()

        results_per_gene[gene_name] = {
            "rank":              rank + 1,
            "mean_expr_rg":      float(rg_mean_expr[pos]),
            "delta_neuroblast":  float(delta_nb),
            "delta_rg":          float(delta_rg),
            "ko_p_neuroblast":   float(ko_probs[:, LEAF_TO_IDX["Neuroblast"]].mean()),
            "ko_p_rg":           float(ko_probs[:, LEAF_TO_IDX["RG"]].mean()),
        }

        if (rank + 1) % 50 == 0:
            elapsed = time.time() - t0
            log(f"  [{rank+1}/{N_TOP_GENES}] 완료 ({elapsed:.0f}s 경과)")

    # ⑥ 결과 정렬: ΔP(Neuroblast) 내림차순
    sorted_genes = sorted(
        results_per_gene.items(),
        key=lambda x: x[1]["delta_neuroblast"],
        reverse=True,
    )

    log(f"\n[5] 결과: RG → Neuroblast 전이 상위 유전자 (KO 시 P(Neuroblast) 증가)")
    log(f"{'유전자':<20} {'ΔP(NB)':>10}  {'ΔP(RG)':>10}  {'P(NB)|KO':>10}  {'Mean Expr':>10}")
    log("-" * 65)
    for gene, info in sorted_genes[:30]:
        log(f"{gene:<20} {info['delta_neuroblast']:>10.4f}  {info['delta_rg']:>10.4f}  "
            f"{info['ko_p_neuroblast']:>10.4f}  {info['mean_expr_rg']:>10.2f}")

    log(f"\n[6] 결과: RG 유지 유전자 (KO 시 P(RG) 감소 → 이 유전자가 RG 정체성 유지)")
    log(f"{'유전자':<20} {'ΔP(RG)':>10}  {'ΔP(NB)':>10}")
    log("-" * 45)
    sorted_by_rg = sorted(
        results_per_gene.items(),
        key=lambda x: x[1]["delta_rg"],
    )
    for gene, info in sorted_by_rg[:20]:
        log(f"{gene:<20} {info['delta_rg']:>10.4f}  {info['delta_neuroblast']:>10.4f}")

    # 결과 저장
    out_data = {
        "model": "scGPT_brain + HCE (step1 finetuned)",
        "n_rg_cells": len(chosen),
        "n_genes_tested": len(results_per_gene),
        "transition": "RG → Neuroblast",
        "baseline": {
            "p_rg":         float(baseline_rg_prob),
            "p_neuroblast": float(baseline_nb_prob),
            "accuracy_rg":  float(baseline_acc),
        },
        "top_transition_genes": [
            {"gene": g, **info}
            for g, info in sorted_genes[:50]
        ],
        "top_rg_identity_genes": [
            {"gene": g, **info}
            for g, info in sorted_by_rg[:30]
        ],
        "all_genes": {g: info for g, info in results_per_gene.items()},
    }
    with open(JSON_PATH, "w") as f:
        json.dump(out_data, f, indent=2)
    log(f"\n결과 저장: {JSON_PATH}")
    log(f"총 소요 시간: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
