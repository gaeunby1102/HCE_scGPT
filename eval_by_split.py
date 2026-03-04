"""
eval_by_split.py
----------------
Norman 데이터셋에서 GEARS+HCE 모델을 split 유형별로 평가.

Split 유형:
  combo_seen0    : 두 유전자 모두 훈련에 없는 이중 섭동 (최고 난이도 OOD)
  combo_seen1    : 한 유전자만 훈련에 있는 이중 섭동
  combo_seen2    : 두 유전자 모두 훈련에 있는 이중 섭동 (최저 난이도)
  unseen_single  : 훈련에 없는 단일 유전자 KO

실행:
    cd /data2/Atlas_Normal
    python -m HCE.eval_by_split
"""

from __future__ import annotations
import sys, os, json
sys.path.insert(0, "/data2/Atlas_Normal")

import numpy as np
import torch
import pickle
from copy import deepcopy
from scipy.stats import pearsonr

from gears import PertData
from gears.utils import print_sys

from HCE.gears_hce import GEARSWithHCE, GEARSModelWithHCE
from HCE.data_replogle import build_k562_go_ontology
from HCE.config import GEARS_DATA_DIR, GEARS_SAVE_NORMAN, RESULTS_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(GEARS_SAVE_NORMAN, "model.pt")
LOG_PATH   = os.path.join(RESULTS_ROOT, "eval_by_split.log")
JSON_PATH  = os.path.join(RESULTS_ROOT, "eval_by_split.json")


def log(msg: str):
    print_sys(msg)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ── 데이터 로딩 ──────────────────────────────────────────────────────────────
def load_data():
    pert_data = PertData(GEARS_DATA_DIR)
    pert_data.load(data_path=os.path.join(GEARS_DATA_DIR, "norman"))
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    return pert_data


# ── 모델 로딩 ────────────────────────────────────────────────────────────────
def load_model(pert_data: PertData):
    """저장된 GEARSWithHCE 모델 로드 (저장된 config로 그래프 재사용)."""
    config_pkl = os.path.join(GEARS_SAVE_NORMAN, "config.pkl")

    # 저장된 config (G_go, G_coexpress 포함) 로드
    with open(config_pkl, "rb") as f:
        saved_config = pickle.load(f)

    log(f"  저장된 config: num_genes={saved_config['num_genes']}, "
        f"hidden_size={saved_config.get('hidden_size', 64)}")

    # GEARSWithHCE 초기화 (ctrl_expression, dict_filter, gene_list, dataloader 등 설정)
    gears = GEARSWithHCE(pert_data, device=DEVICE)

    # 저장된 그래프를 전달해서 model_initialize_hce 호출
    gears.model_initialize_hce(
        hidden_size=saved_config.get("hidden_size", 64),
        num_go_gnn_layers=saved_config.get("num_go_gnn_layers", 1),
        num_gene_gnn_layers=saved_config.get("num_gene_gnn_layers", 1),
        decoder_hidden_size=saved_config.get("decoder_hidden_size", 16),
        num_similar_genes_go_graph=saved_config.get("num_similar_genes_go_graph", 20),
        num_similar_genes_co_express_graph=saved_config.get("num_similar_genes_co_express_graph", 20),
        coexpress_threshold=saved_config.get("coexpress_threshold", 0.4),
        direction_lambda=saved_config.get("direction_lambda", 0.1),
        lambda_hce=0.3,
        G_go=saved_config.get("G_go"),
        G_go_weight=saved_config.get("G_go_weight"),
        G_coexpress=saved_config.get("G_coexpress"),
        G_coexpress_weight=saved_config.get("G_coexpress_weight"),
    )

    # pert_emb 크기 불일치 해결: 저장된 크기로 config 수정 후 모델 재생성
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    n_perts_saved = state["pert_emb.weight"].shape[0]
    if gears.model.pert_emb.weight.shape[0] != n_perts_saved:
        log(f"  pert_emb 재설정: {gears.model.pert_emb.weight.shape[0]} → {n_perts_saved}")
        gears.config["num_perts"] = n_perts_saved
        n_go = len(gears.hce_term_to_idx)
        gears.model = GEARSModelWithHCE(gears.config, n_go=n_go).to(DEVICE)

    gears.model.load_state_dict(state)
    gears.best_model = deepcopy(gears.model)
    log(f"모델 로드 완료: {MODEL_PATH}")
    return gears


# ── 섭동 단위 평가 ────────────────────────────────────────────────────────────
def evaluate_loader(model, loader, device) -> dict:
    """
    loader의 모든 배치를 평가해 섭동별 Pearson 반환.
    Returns: {pert_name: pearson_r}
    """
    model.eval()
    results: dict[str, list] = {}
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            pred, _ = model(batch)
            y      = batch.y
            perts  = np.array(batch.pert)
            for p in set(perts):
                idx    = np.where(perts == p)[0]
                p_pred = pred[idx].cpu().numpy()
                p_true = y[idx].cpu().numpy()
                r_vals = [pearsonr(p_pred[i], p_true[i])[0] for i in range(len(idx))]
                results.setdefault(p, []).extend(r_vals)
    return {p: float(np.nanmean(v)) for p, v in results.items()}


# ── 분할별 집계 ───────────────────────────────────────────────────────────────
def aggregate_by_subgroup(
    per_pert: dict[str, float],
    subgroup: dict[str, list],
) -> dict[str, dict]:
    """subgroup 딕셔너리를 이용해 split 유형별 통계 계산."""
    out = {}
    for split_name, conds in subgroup.items():
        vals = [per_pert[c] for c in conds if c in per_pert]
        if not vals:
            out[split_name] = {"n": 0, "mean_pearson": None, "std_pearson": None}
            continue
        out[split_name] = {
            "n":            len(vals),
            "mean_pearson": float(np.nanmean(vals)),
            "std_pearson":  float(np.nanstd(vals)),
            "min_pearson":  float(np.nanmin(vals)),
            "max_pearson":  float(np.nanmax(vals)),
        }
    return out


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("GEARS+HCE split-type 평가 시작")
    log("=" * 60)

    # 데이터 로딩
    log("Norman 데이터 로딩...")
    pert_data = load_data()
    test_loader = pert_data.dataloader["test_loader"]
    subgroup    = pert_data.subgroup["test_subgroup"]

    log("Split 구성:")
    for k, v in subgroup.items():
        log(f"  {k}: {len(v)} conditions")

    # 모델 로딩
    gears = load_model(pert_data)

    # Test set 전체 평가
    log("\nTest set 전체 평가 중...")
    per_pert = evaluate_loader(gears.model, test_loader, DEVICE)
    overall  = float(np.nanmean(list(per_pert.values())))
    log(f"Overall test Pearson: {overall:.4f}  (n={len(per_pert)} conditions)")

    # Split 유형별 집계
    by_split = aggregate_by_subgroup(per_pert, subgroup)

    log("\n[ Split 유형별 결과 ]")
    log(f"{'Split':<20} {'N':>4}  {'Mean Pearson':>13}  {'Std':>7}")
    log("-" * 50)
    for split_name in ["combo_seen0", "combo_seen1", "combo_seen2", "unseen_single"]:
        r = by_split.get(split_name, {})
        if r.get("n", 0) == 0:
            log(f"{split_name:<20}  ---   (조건 없음)")
        else:
            log(f"{split_name:<20} {r['n']:>4}  {r['mean_pearson']:>13.4f}  {r['std_pearson']:>7.4f}")

    # 각 섭동 결과 상세 출력 (combo_seen0: 가장 어려운 케이스)
    log("\n[ combo_seen0 개별 섭동 결과 (n=9) ]")
    for cond in subgroup.get("combo_seen0", []):
        r_val = per_pert.get(cond, float("nan"))
        log(f"  {cond:<35} Pearson = {r_val:.4f}")

    log("\n[ unseen_single 개별 섭동 결과 ]")
    single_vals = [(c, per_pert.get(c, float("nan"))) for c in subgroup.get("unseen_single", [])]
    single_vals.sort(key=lambda x: x[1], reverse=True)
    for cond, r_val in single_vals[:10]:
        log(f"  {cond:<35} Pearson = {r_val:.4f}")

    # 결과 저장
    out_data = {
        "model": "GEARS+HCE (λ=0.3)",
        "overall_test_pearson": overall,
        "n_conditions": len(per_pert),
        "by_split": by_split,
        "per_perturbation": per_pert,
    }
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(out_data, f, indent=2)
    log(f"\n결과 저장: {JSON_PATH}")


if __name__ == "__main__":
    main()
