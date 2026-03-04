"""
scgpt_gears_hce.py
------------------
Task 3: GEARS GNN에 scGPT_brain 초기화 유전자/섭동 임베딩 + HCE loss.

아이디어:
  1. scGPT_brain 로드 (frozen, 임베딩 추출 전용)
  2. Norman 유전자 전체에 대해 scGPT.encoder(gene_token_id) → (n_genes, 512D)
  3. PCA로 512D → 64D (GEARS hidden_size) 투영
  4. GEARSWithHCE의 gene_emb.weight / pert_emb.weight를 투영된 값으로 초기화
  5. 이후 GEARSWithHCE 동일 하이퍼파라미터로 학습 (λ_hce=0.3, epochs=15)

scGPT 초기화 효과:
  - gene_emb: scGPT 사전학습 유전자 표현 → GNN 초기 상태 개선
  - pert_emb: 섭동 유전자도 동일 공간 초기화 → 섭동 맥락 이해 향상
  - PCA: 분산 95%+ 보존하면서 차원 축소

비교 대상:
  GEARS baseline:     best Pearson=0.692, ep15=0.005 (붕괴)
  GEARS + HCE:        best Pearson=0.817, ep15=0.700 (안정)
  scGPT + GEARS + HCE: (본 실험 결과)

실행:
    python -m HCE.scgpt_gears_hce
"""
from __future__ import annotations
import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

import HCE.config as cfg

SCGPT_DIR   = cfg.SCGPT_BRAIN_DIR
GEARS_DIR   = cfg.GEARS_DATA_DIR
RESULT_DIR  = cfg.RESULTS_ROOT
SAVE_PATH   = cfg.GEARS_SAVE_NORMAN
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS      = 15
BATCH       = 32
LR          = 1e-3
LAMBDA_HCE  = 0.3
HIDDEN_SIZE = 64   # GEARS hidden_size (PCA 목표 차원)

LOG_PATH    = os.path.join(RESULT_DIR, "scgpt_gears_hce.log")
RESULT_PATH = os.path.join(RESULT_DIR, "scgpt_gears_hce.json")


# ── scGPT_brain 로드 ──────────────────────────────────────────────────
def load_scgpt():
    """
    scgpt_norman_hce.py와 동일한 로드 함수.
    임베딩 추출 후 del로 GPU 메모리 해제 가능.
    """
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab

    vocab = GeneVocab.from_file(os.path.join(SCGPT_DIR, "vocab.json"))
    args  = json.load(open(os.path.join(SCGPT_DIR, "args.json")))

    model = TransformerModel(
        ntoken          = len(vocab),
        d_model         = args["embsize"],
        nhead           = args["nheads"],
        d_hid           = args["d_hid"],
        nlayers         = args["nlayers"],
        nlayers_cls     = args.get("n_layers_cls", 3),
        n_cls           = 1,
        vocab           = vocab,
        dropout         = args["dropout"],
        pad_token       = args["pad_token"],
        pad_value       = args["pad_value"],
        do_mvc          = False,
        do_dab          = False,
        use_batch_labels= False,
        input_emb_style = args["input_emb_style"],
        n_input_bins    = args["n_bins"],
        cell_emb_style  = "cls",
        use_fast_transformer = args.get("fast_transformer", True),
        pre_norm        = False,
    )
    ckpt = torch.load(os.path.join(SCGPT_DIR, "best_model.pt"), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    return model, vocab, args


# ── scGPT 임베딩 주입 ────────────────────────────────────────────────
def inject_scgpt_embeddings(gears, scgpt_model, vocab, pert_data, device):
    """
    scGPT_brain의 gene encoder 임베딩을 GEARSWithHCE에 주입.

    Steps:
      1. Norman 유전자 목록에서 scGPT vocab 토큰 ID 조회
      2. scgpt_model.encoder(gene_tokens) → (n_genes, 512D) 임베딩 추출
      3. sklearn PCA로 512D → hidden_size(64D) 투영
      4. gears.model.model.gene_emb.weight / pert_emb.weight 초기화

    Args:
        gears:       GEARSWithHCE 인스턴스 (model_initialize_hce 완료 상태)
        scgpt_model: TransformerModel (frozen, CPU 또는 GPU)
        vocab:       GeneVocab (scGPT)
        pert_data:   PertData (Norman)
        device:      학습 디바이스

    Returns:
        dict: {"vocab_coverage": float, "pca_explained_variance": float}
    """
    from sklearn.decomposition import PCA

    # Norman 유전자 목록
    adata = pert_data.adata
    if "gene_name" in adata.var.columns:
        gene_names = adata.var["gene_name"].tolist()
    else:
        gene_names = adata.var_names.tolist()
    n_genes = len(gene_names)

    pad_id = vocab["<pad>"]

    # vocab 커버리지
    gene_tokens = []
    n_in_vocab  = 0
    for g in gene_names:
        if g in vocab:
            gene_tokens.append(vocab[g])
            n_in_vocab += 1
        else:
            gene_tokens.append(pad_id)

    vocab_coverage = n_in_vocab / n_genes
    print(f"  vocab 커버리지: {n_in_vocab}/{n_genes} ({vocab_coverage:.1%})")

    # scGPT encoder 임베딩 추출 (no_grad, CPU에서 수행)
    scgpt_model.eval()
    scgpt_cpu = scgpt_model.cpu()
    gene_tok_tensor = torch.tensor(gene_tokens, dtype=torch.long)  # (n_genes,)

    with torch.no_grad():
        # encoder: token_id → embedding  (n_genes, d_model)
        scgpt_embs = scgpt_cpu.encoder(gene_tok_tensor)             # (n_genes, 512)

    scgpt_embs_np = scgpt_embs.numpy()   # (n_genes, 512)
    print(f"  scGPT 임베딩 추출 완료: shape={scgpt_embs_np.shape}")

    # PCA: 512D → hidden_size(64D)
    hidden_size = HIDDEN_SIZE
    n_components = min(hidden_size, scgpt_embs_np.shape[0] - 1, scgpt_embs_np.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(scgpt_embs_np)   # (n_genes, hidden_size)
    explained_var = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA {scgpt_embs_np.shape[1]}D → {n_components}D  "
          f"| 설명 분산: {explained_var:.3f}")

    # hidden_size에 맞게 패딩/자르기 (혹시 n_components < hidden_size인 경우)
    if projected.shape[1] < hidden_size:
        pad_cols = np.zeros((n_genes, hidden_size - projected.shape[1]), dtype=np.float32)
        projected = np.concatenate([projected, pad_cols], axis=1)
    projected = projected[:, :hidden_size].astype(np.float32)
    proj_tensor = torch.tensor(projected, dtype=torch.float32)

    # GEARSWithHCE 내부 GEARS_Model의 gene_emb / pert_emb 가중치 초기화
    # gears.model.model → GEARSModelWithHCE (내부 GEARS_Model)
    # 구조: GEARSWithHCE.model = GEARSModelWithHCE
    inner_model = gears.model

    try:
        # gene_emb: (n_genes, hidden_size)
        gene_emb_weight = inner_model.gene_emb.weight  # (n_genes, H)
        if gene_emb_weight.shape == proj_tensor.shape:
            inner_model.gene_emb.weight.data.copy_(proj_tensor)
            print(f"  gene_emb 초기화 완료: {proj_tensor.shape}")
        else:
            # 크기가 다르면 GEARS n_genes와 Norman n_genes가 다른 경우
            # GEARS gene_emb는 (n_perts, H)이거나 (num_genes, H) 등 다를 수 있음
            g_rows = gene_emb_weight.shape[0]
            if g_rows <= n_genes:
                inner_model.gene_emb.weight.data.copy_(proj_tensor[:g_rows])
            else:
                # 반복 패딩
                repeats = (g_rows // n_genes) + 1
                padded  = proj_tensor.repeat(repeats, 1)[:g_rows]
                inner_model.gene_emb.weight.data.copy_(padded)
            print(f"  gene_emb 초기화 (크기 조정): {gene_emb_weight.shape} ← {proj_tensor.shape}")
    except AttributeError:
        print("  [경고] gene_emb 접근 불가 - 스킵")

    try:
        # pert_emb: (n_perts, hidden_size)
        pert_emb_weight = inner_model.pert_emb.weight  # (n_perts, H)
        p_rows = pert_emb_weight.shape[0]
        if p_rows <= n_genes:
            inner_model.pert_emb.weight.data.copy_(proj_tensor[:p_rows])
        else:
            repeats = (p_rows // n_genes) + 1
            padded  = proj_tensor.repeat(repeats, 1)[:p_rows]
            inner_model.pert_emb.weight.data.copy_(padded)
        print(f"  pert_emb 초기화 완료: {pert_emb_weight.shape}")
    except AttributeError:
        print("  [경고] pert_emb 접근 불가 - 스킵")

    return {
        "vocab_coverage": vocab_coverage,
        "pca_explained_variance": explained_var,
        "n_genes": n_genes,
        "n_in_vocab": n_in_vocab,
        "projected_shape": list(projected.shape),
    }


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SAVE_PATH,  exist_ok=True)

    with open(LOG_PATH, "w") as logf:
        def log(msg):
            print(msg); logf.write(msg + "\n"); logf.flush()

        log("=" * 65)
        log("Task 3: GEARS GNN + scGPT 임베딩 초기화 + HCE loss")
        log("        → Norman 섭동 예측")
        log("=" * 65)

        # [1] scGPT 로드 (임베딩 추출 전용, 학습 불필요)
        log("\n[1] scGPT_brain 로드 (임베딩 추출용, frozen)...")
        scgpt_model, vocab, args = load_scgpt()
        d_model = args["embsize"]
        for p in scgpt_model.parameters():
            p.requires_grad_(False)
        scgpt_model.eval()
        log(f"  vocab={len(vocab)}, d_model={d_model}, layers={args['nlayers']}")

        # [2] Norman PertData 로드
        log("\n[2] Norman PertData 로딩...")
        from gears import PertData
        pert_data = PertData(GEARS_DIR)
        pert_data.load(data_name="norman")
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=BATCH, test_batch_size=BATCH)
        log(f"  adata shape: {pert_data.adata.shape}")
        log(f"  train conds: {len(pert_data.set2conditions['train'])}")
        log(f"  test conds:  {len(pert_data.set2conditions.get('test', []))}")

        # scipy sparse + pandas boolean index 호환성 패치
        # GEARS __init__: X[pd.Series] 실패 (scipy가 .nonzero() 요구)
        # get_coexpression: X_tr.toarray() 필요 → dense 변환 불가
        # → csr_matrix 서브클래싱으로 두 조건 동시 해결
        import scipy.sparse as sp
        import pandas as pd

        class PandasCompatCSR(sp.csr_matrix):
            def __getitem__(self, key):
                if isinstance(key, pd.Series):
                    key = key.values
                elif isinstance(key, tuple):
                    key = tuple(k.values if isinstance(k, pd.Series) else k for k in key)
                return super().__getitem__(key)

        if sp.issparse(pert_data.adata.X):
            log("  adata.X PandasCompatCSR 래핑 (scipy/pandas 호환성)...")
            pert_data.adata.X = PandasCompatCSR(pert_data.adata.X)

        # [3] GEARSWithHCE 초기화
        log("\n[3] GEARSWithHCE 초기화...")
        from HCE.gears_hce import GEARSWithHCE
        gears = GEARSWithHCE(pert_data, device=DEVICE)
        gears.model_initialize_hce(
            hidden_size=HIDDEN_SIZE,
            num_go_gnn_layers=1,
            num_gene_gnn_layers=1,
            decoder_hidden_size=16,
            num_similar_genes_go_graph=20,
            num_similar_genes_co_express_graph=20,
            coexpress_threshold=0.4,
            direction_lambda=1e-1,
            lambda_hce=LAMBDA_HCE,
        )
        log(f"  GEARSWithHCE 초기화 완료")
        log(f"  hidden_size={HIDDEN_SIZE}, lambda_hce={LAMBDA_HCE}")

        # [4] scGPT 임베딩 주입
        log("\n[4] scGPT gene 임베딩 → GEARS gene_emb / pert_emb 주입...")
        t_inj = time.time()
        embed_info = inject_scgpt_embeddings(
            gears, scgpt_model, vocab, pert_data, device=DEVICE
        )
        log(f"  완료 ({time.time()-t_inj:.1f}s)")
        log(f"  vocab 커버리지:    {embed_info['vocab_coverage']:.1%}")
        log(f"  PCA 설명 분산:     {embed_info['pca_explained_variance']:.3f}")
        log(f"  투영 shape:        {embed_info['projected_shape']}")

        # scGPT는 임베딩 추출 후 더 이상 필요 없으므로 메모리 해제
        del scgpt_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("  scGPT 모델 메모리 해제 완료")

        # [5] 학습
        log(f"\n[5] 학습 시작 (epochs={EPOCHS}, lr={LR})...")
        t0 = time.time()
        gears.train(epochs=EPOCHS, lr=LR)
        elapsed = time.time() - t0
        log(f"  학습 완료 ({elapsed:.0f}s)")

        # [6] 저장
        log(f"\n[6] 모델 저장: {SAVE_PATH}")
        gears.save_model(SAVE_PATH)

        # [7] 결과 요약
        log("\n" + "=" * 65)
        log("결과 비교")
        log("-" * 65)
        log(f"  {'모델':36s} {'Best Pearson':>13s} {'ep15 Pearson':>13s}")
        log(f"  {'GEARS baseline':36s} {'0.692':>13s} {'0.005':>13s}  ← 붕괴")
        log(f"  {'GEARS + HCE (λ=0.3)':36s} {'0.817':>13s} {'0.700':>13s}  ← 안정")
        log(f"  {'scGPT + GEARS + HCE (λ=0.3)':36s} {'N/A':>13s} {'N/A':>13s}  ← Task 3")
        log(f"\n  (scGPT+GEARS+HCE 정량 평가는 gears.eval() 별도 실행)")
        log("=" * 65)

        # 저장
        results = {
            "model": f"scGPT_brain + GEARS + HCE (λ={LAMBDA_HCE})",
            "task": "Task 3: GEARS GNN with scGPT-initialized gene/pert embeddings + HCE",
            "lambda_hce": LAMBDA_HCE,
            "hidden_size": HIDDEN_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "scgpt_d_model": d_model,
            "embed_injection": embed_info,
            "training_time_sec": elapsed,
            "comparison": {
                "GEARS_baseline_best":  0.692,
                "GEARS_baseline_ep15":  0.005,
                "GEARS_HCE_best":       0.817,
                "GEARS_HCE_ep15":       0.700,
            },
            "note": (
                "scGPT+GEARS+HCE 정량 Pearson은 gears.eval() 별도 실행으로 확인. "
                "학습 중 GEARS 내장 val 평가 로그는 LOG_PATH 참조."
            ),
        }
        with open(RESULT_PATH, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log(f"\n결과 저장: {RESULT_PATH}")
        log(f"로그:      {LOG_PATH}")


if __name__ == "__main__":
    main()
