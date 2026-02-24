"""
gears_norman_baseline.py
------------------------
GEARS (HCE 없이) Norman 데이터 학습 → GEARSWithHCE (λ=0.3)과 비교.

실행:
    python -m HCE.gears_norman_baseline
"""
from __future__ import annotations
import sys, os, json
import warnings; warnings.filterwarnings("ignore")

from gears import PertData, GEARS
import HCE.config as cfg

DATA_PATH   = cfg.GEARS_DATA_DIR
SAVE_PATH   = cfg.GEARS_SAVE_BASELINE
RESULT_PATH = os.path.join(cfg.RESULTS_ROOT, "norman_comparison.json")


def main():
    print("=" * 60)
    print("GEARS Baseline: Norman (λ_hce=0, 비교용)")
    print("=" * 60)

    print("\n[1] Norman PertData 로딩...")
    pert_data = PertData(DATA_PATH)
    pert_data.load(data_name="norman")
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=32, test_batch_size=64)
    print(f"  adata: {pert_data.adata.shape}, "
          f"train: {len(pert_data.set2conditions['train'])}")

    print("\n[2] GEARS 초기화...")
    gears = GEARS(pert_data, device="cuda")
    gears.model_initialize(
        hidden_size=64,
        num_go_gnn_layers=1,
        num_gene_gnn_layers=1,
        decoder_hidden_size=16,
        num_similar_genes_go_graph=20,
        num_similar_genes_co_express_graph=20,
        coexpress_threshold=0.4,
        direction_lambda=1e-1,
    )

    print("\n[3] 학습 (epochs=15)...")
    gears.train(epochs=15, lr=1e-3)

    # GEARS는 내부적으로 val 출력 — 로그에서 파싱
    print("\n[4] 평가...")
    test_res = gears.predict(
        [[p.split("+")[0]] for p in list(pert_data.set2conditions.get("test", []))[:5]
         if "ctrl" in p]
    )

    print("\n학습 완료.")

    # 저장
    os.makedirs(SAVE_PATH, exist_ok=True)
    gears.save_model(SAVE_PATH)

    # HCE 결과와 비교
    hce_log_path = os.path.join(cfg.RESULTS_ROOT, "norman_hce.log")
    hce_results = {}
    if os.path.exists(hce_log_path):
        with open(hce_log_path) as f:
            for line in f:
                if "Val Pearson" in line:
                    parts = line.strip().split("|")
                    ep = int(parts[0].split()[1].rstrip(":"))
                    mse = float(parts[1].split("=")[1])
                    pearson = float(parts[2].split("=")[1])
                    hce_results[ep] = {"mse": mse, "pearson": pearson}

    print("\n[HCE 학습 요약 (λ=0.3)]")
    if hce_results:
        best_ep = max(hce_results, key=lambda e: hce_results[e]["pearson"])
        print(f"  Best Epoch {best_ep}: "
              f"Pearson={hce_results[best_ep]['pearson']:.4f}, "
              f"MSE={hce_results[best_ep]['mse']:.4f}")
        if 15 in hce_results:
            print(f"  Final Epoch 15: "
                  f"Pearson={hce_results[15]['pearson']:.4f}, "
                  f"MSE={hce_results[15]['mse']:.4f}")

    result = {
        "hce_lambda0.3": hce_results,
        "note": "GEARS baseline val metrics are printed during training; "
                "see gears_baseline_norman/ for saved model"
    }
    os.makedirs(cfg.RESULTS_ROOT, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
