"""
gears_norman_hce.py
-------------------
Norman 데이터 + PertData 파이프라인으로 GEARSWithHCE 전체 학습.

실행:
    /home/t1/miniconda3/envs/gears2/bin/python -m HCE.gears_norman_hce
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, "/data2/Atlas_Normal")

import warnings
warnings.filterwarnings("ignore")

from gears import PertData
from HCE.gears_hce import GEARSWithHCE

DATA_PATH   = "/data4/HCE_gears_data"
DEVICE      = "cuda"
SAVE_PATH   = "/data4/HCE_gears_data/gears_hce_norman"


def main():
    print("=" * 60)
    print("GEARS + HCE: Norman 데이터 전체 학습")
    print("=" * 60)

    # ── 데이터 로딩 ──────────────────────────────────────────────
    print("\n[1] Norman PertData 로딩...")
    pert_data = PertData(DATA_PATH)
    pert_data.load(data_name="norman")
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=32, test_batch_size=64)

    print(f"  adata shape: {pert_data.adata.shape}")
    print(f"  train conds: {len(pert_data.set2conditions['train'])}")
    print(f"  test conds:  {len(pert_data.set2conditions.get('test', []))}")

    # ── 모델 초기화 ──────────────────────────────────────────────
    print("\n[2] GEARSWithHCE 초기화...")
    gears = GEARSWithHCE(pert_data, device=DEVICE)
    gears.model_initialize_hce(
        hidden_size=64,
        num_go_gnn_layers=1,
        num_gene_gnn_layers=1,
        decoder_hidden_size=16,
        num_similar_genes_go_graph=20,
        num_similar_genes_co_express_graph=20,
        coexpress_threshold=0.4,
        direction_lambda=1e-1,
        lambda_hce=0.3,
    )

    # ── 학습 ─────────────────────────────────────────────────────
    print("\n[3] 학습 시작 (epochs=15)...")
    gears.train(epochs=15, lr=1e-3)

    # ── 저장 ─────────────────────────────────────────────────────
    os.makedirs(SAVE_PATH, exist_ok=True)
    gears.save_model(SAVE_PATH)
    print(f"\n[4] 모델 저장: {SAVE_PATH}")

    # ── 예측 테스트 ───────────────────────────────────────────────
    print("\n[5] 예측 테스트...")
    test_perts = list(pert_data.set2conditions.get("test", []))[:3]
    if test_perts:
        parsed = [[p.split("+")[0]] for p in test_perts if "ctrl" in p][:3]
        if parsed:
            preds = gears.predict(parsed)
            for pert, pred in preds.items():
                print(f"  {pert}: pred shape={pred.shape}, mean={pred.mean():.4f}")


if __name__ == "__main__":
    main()
