"""
interaction_analysis.py
-----------------------
Norman 이중 섭동 상호작용 분석.

핵심 질문:
  실제 Δexpr(A+B) vs. Δexpr(A) + Δexpr(B) 차이 → 시너지/길항 작용
  GEARS+HCE가 이 상호작용을 얼마나 잘 예측하는가?

분석 대상:
  combo_seen2 (18개): 두 유전자 모두 훈련 포함 → 단일 효과 참조 가능
  combo_seen1 (52개): 한 유전자만 포함 → 부분 참조
  combo_seen0 (9개):  둘 다 미포함 → 시너지 예측 한계 케이스

실행:
    cd /data2/Atlas_Normal
    conda run -n gears2 python -m HCE.interaction_analysis
"""

from __future__ import annotations
import sys, os, json, pickle
sys.path.insert(0, "/data2/Atlas_Normal")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from copy import deepcopy

from gears import PertData
from gears.utils import print_sys

from HCE.gears_hce import GEARSWithHCE, GEARSModelWithHCE
from HCE.data_replogle import build_k562_go_ontology
from HCE.config import GEARS_DATA_DIR, GEARS_SAVE_NORMAN, RESULTS_ROOT

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(GEARS_SAVE_NORMAN, "model.pt")
LOG_PATH   = os.path.join(RESULTS_ROOT, "interaction_analysis.log")
JSON_PATH  = os.path.join(RESULTS_ROOT, "interaction_analysis.json")
FIG_DIR    = os.path.join(RESULTS_ROOT, "figures")


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


# ── 모델 로딩 ─────────────────────────────────────────────────────────────────
def load_model(pert_data):
    config_pkl = os.path.join(GEARS_SAVE_NORMAN, "config.pkl")
    with open(config_pkl, "rb") as f:
        saved_config = pickle.load(f)

    gears = GEARSWithHCE(pert_data, device=DEVICE)
    gears.model_initialize_hce(
        hidden_size=saved_config.get("hidden_size", 64),
        direction_lambda=saved_config.get("direction_lambda", 0.1),
        lambda_hce=0.3,
        G_go=saved_config.get("G_go"),
        G_go_weight=saved_config.get("G_go_weight"),
        G_coexpress=saved_config.get("G_coexpress"),
        G_coexpress_weight=saved_config.get("G_coexpress_weight"),
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    n_perts_saved = state["pert_emb.weight"].shape[0]
    if gears.model.pert_emb.weight.shape[0] != n_perts_saved:
        gears.config["num_perts"] = n_perts_saved
        n_go = len(gears.hce_term_to_idx)
        gears.model = GEARSModelWithHCE(gears.config, n_go=n_go).to(DEVICE)
    gears.model.load_state_dict(state)
    gears.best_model = deepcopy(gears.model)
    log(f"모델 로드 완료: {MODEL_PATH}")
    return gears


# ── adata에서 조건별 mean 발현값 사전 계산 ──────────────────────────────────
def build_actual_delta_cache(pert_data) -> dict[str, np.ndarray]:
    """모든 조건의 mean(X) - ctrl_mean 사전 계산."""
    adata = pert_data.adata
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)

    ctrl_mask = adata.obs["condition"] == "ctrl"
    ctrl_mean = X[ctrl_mask].mean(axis=0)  # (n_genes,)

    cache = {}
    for cond in adata.obs["condition"].unique():
        mask = adata.obs["condition"] == cond
        cache[cond] = X[mask].mean(axis=0) - ctrl_mean
    log(f"  실제 발현 캐시 완료: {len(cache)} 조건")
    return cache


# ── 전체 loader에서 조건별 예측 사전 계산 ────────────────────────────────────
@torch.no_grad()
def build_pred_cache(gears, pert_data) -> dict[str, np.ndarray]:
    """
    test/val/train loader 전체 순회 → 조건별 mean 예측 Δexpr 캐시.
    GEARSModelWithHCE.forward() → (pred, go_logits) 튜플 처리.
    """
    gears.model.eval()
    cache_sum   = {}
    cache_count = {}

    for loader_key in ["test_loader", "val_loader", "train_loader"]:
        loader = pert_data.dataloader.get(loader_key)
        if loader is None:
            continue
        log(f"  예측 캐시: {loader_key} 처리 중...")
        for batch in loader:
            batch.to(DEVICE)
            pred_out, _ = gears.model(batch)          # (B*G,) or (B, G)
            # pred_out을 (B, G) 형태로 reshape
            n_genes = gears.num_genes
            pred = pred_out.reshape(-1, n_genes).cpu().numpy()  # (B, G)
            perts = np.array(batch.pert)
            unique_perts = set(perts)
            for p in unique_perts:
                idx = np.where(perts == p)[0]
                arr = pred[idx].sum(axis=0)
                cnt = len(idx)
                if p not in cache_sum:
                    cache_sum[p]   = arr
                    cache_count[p] = cnt
                else:
                    cache_sum[p]   += arr
                    cache_count[p] += cnt

    # 평균화 + ctrl 기준 차이 계산
    # (GEARSWithHCE는 직접 Δexpr를 출력하므로 그대로 사용)
    pred_cache = {
        p: cache_sum[p] / cache_count[p]
        for p in cache_sum
    }
    log(f"  예측 캐시 완료: {len(pred_cache)} 조건")
    return pred_cache


# ── 상호작용 계산 ─────────────────────────────────────────────────────────────
def interaction_score(delta_AB, delta_A, delta_B):
    """
    interaction = Δ(A+B) - [Δ(A) + Δ(B)]
    > 0: 시너지, < 0: 길항
    """
    return delta_AB - (delta_A + delta_B)


# ── 분석 메인 ─────────────────────────────────────────────────────────────────
def analyze(gears, pert_data, actual_cache, pred_cache) -> dict:
    sg = pert_data.subgroup
    all_conds = set(actual_cache.keys())
    results = {}

    for split_name in ["combo_seen2", "combo_seen1", "combo_seen0"]:
        combos = list(sg["test_subgroup"].get(split_name, []))
        log(f"\n--- {split_name} ({len(combos)} conditions) ---")
        split_results = []

        for cond in combos:
            g1, g2 = cond.split("+")

            # 단일 섭동 조건 이름 탐색
            single_A = next((c for c in [f"{g1}+ctrl", f"ctrl+{g1}"] if c in all_conds), None)
            single_B = next((c for c in [f"{g2}+ctrl", f"ctrl+{g2}"] if c in all_conds), None)

            # ── 실제 발현 변화 (adata 캐시에서) ──
            actual_AB = actual_cache.get(cond)
            actual_A  = actual_cache.get(single_A) if single_A else None
            actual_B  = actual_cache.get(single_B) if single_B else None

            if actual_AB is None:
                log(f"  [SKIP] {cond}: adata에 없음")
                continue

            # ── 모델 예측 (pred_cache에서) ──
            pred_AB = pred_cache.get(cond)
            pred_A  = pred_cache.get(single_A) if single_A else None
            pred_B  = pred_cache.get(single_B) if single_B else None

            if pred_AB is None:
                log(f"  [SKIP] {cond}: 모델 예측 실패")
                continue

            entry = {
                "condition": cond,
                "g1": g1, "g2": g2,
                "has_single_A": single_A is not None,
                "has_single_B": single_B is not None,
            }

            # 예측 정확도
            r_pred, _ = pearsonr(pred_AB, actual_AB)
            entry["pearson_combo"] = float(r_pred)

            # 상호작용 분석 (단일 데이터가 있을 때)
            if actual_A is not None and actual_B is not None:
                actual_inter = interaction_score(actual_AB, actual_A, actual_B)
                n_syn = int((actual_inter > 0).sum())
                n_ant = int((actual_inter < 0).sum())
                n_tot = n_syn + n_ant
                entry["actual_synergy_magnitude"]  = float(np.abs(actual_inter).mean())
                entry["actual_synergy_ratio"]       = float(n_syn / (n_tot + 1e-8))
                entry["actual_synergistic_genes"]   = n_syn
                entry["actual_antagonistic_genes"]  = n_ant

                # 단순합산(additive) baseline Pearson
                actual_add = actual_A + actual_B
                r_add, _   = pearsonr(actual_add, actual_AB)
                entry["pearson_additive_baseline"]  = float(r_add)

                # 예측 상호작용 vs 실제 상호작용
                if pred_A is not None and pred_B is not None:
                    pred_inter = interaction_score(pred_AB, pred_A, pred_B)
                    r_int, _   = pearsonr(pred_inter, actual_inter)
                    entry["pearson_interaction"]      = float(r_int)
                    entry["pred_synergy_magnitude"]   = float(np.abs(pred_inter).mean())
                    n_pred_syn = int((pred_inter > 0).sum())
                    entry["pred_synergy_ratio"]       = float(n_pred_syn / (len(pred_inter) + 1e-8))

                    direction = "시너지" if n_syn > n_ant else "길항"
                    log(f"  {cond}:")
                    log(f"    Pearson(모델 vs 실제)     = {r_pred:.4f}")
                    log(f"    Pearson(단순합산 baseline)= {r_add:.4f}")
                    log(f"    Pearson(상호작용 예측)    = {r_int:.4f}")
                    log(f"    실제 {direction}: syn={n_syn}, ant={n_ant}, "
                        f"ratio={n_syn/n_tot:.2f}")
                else:
                    log(f"  {cond}: Pearson={r_pred:.4f} | additive={r_add:.4f} "
                        f"(단일 예측 불가)")
            else:
                log(f"  {cond}: Pearson={r_pred:.4f} "
                    f"(단일 데이터 없음: A={single_A is not None}, B={single_B is not None})")

            split_results.append(entry)

        results[split_name] = split_results

    return results


# ── 시각화 ───────────────────────────────────────────────────────────────────
def visualize(results: dict):
    os.makedirs(FIG_DIR, exist_ok=True)

    seen2 = [r for r in results.get("combo_seen2", []) if "pearson_interaction" in r]
    if not seen2:
        log("시각화 데이터 없음")
        return

    conds      = [r["condition"] for r in seen2]
    r_pred     = [r["pearson_combo"] for r in seen2]
    r_add      = [r["pearson_additive_baseline"] for r in seen2]
    r_int      = [r["pearson_interaction"] for r in seen2]
    syn_ratio  = [r["actual_synergy_ratio"] for r in seen2]
    syn_mag    = [r["actual_synergy_magnitude"] for r in seen2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Norman combo_seen2: GEARS+HCE 이중 섭동 상호작용 분석", fontsize=13)

    # (a) Model Pearson vs Additive Baseline Pearson
    ax = axes[0]
    sc = ax.scatter(r_add, r_pred, c=syn_ratio, cmap="RdBu_r", vmin=0, vmax=1,
                    s=100, edgecolors="gray", linewidths=0.5, zorder=3)
    lo = min(min(r_add), min(r_pred)) - 0.05
    hi = max(max(r_add), max(r_pred)) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="y=x (동등)")
    # 모델이 앞서는 영역
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.05, color="green")
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], alpha=0.05, color="red")
    ax.set_xlabel("단순합산 Baseline Pearson", fontsize=11)
    ax.set_ylabel("GEARS+HCE Pearson", fontsize=11)
    ax.set_title("(a) 모델 vs 단순합산\n(위쪽 = 모델 우세)", fontsize=11)
    plt.colorbar(sc, ax=ax, label="시너지 비율")
    for i, c in enumerate(conds):
        ax.annotate(c, (r_add[i], r_pred[i]), fontsize=5.5,
                    ha="center", va="bottom", alpha=0.8)
    ax.legend(fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # (b) 상호작용 예측 Pearson (바차트)
    ax2 = axes[1]
    sorted_idx = np.argsort(r_int)
    y_pos = np.arange(len(sorted_idx))
    colors_bar = ["#27ae60" if r_int[i] > 0 else "#e74c3c" for i in sorted_idx]
    ax2.barh(y_pos, [r_int[i] for i in sorted_idx], color=colors_bar, edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([conds[i] for i in sorted_idx], fontsize=7)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Pearson(예측 상호작용 vs 실제 상호작용)", fontsize=10)
    ax2.set_title("(b) 상호작용 예측 정확도\n(양수 = 시너지/길항 패턴 포착)", fontsize=10)
    green_p = mpatches.Patch(color="#27ae60", label="r > 0 (포착)")
    red_p   = mpatches.Patch(color="#e74c3c", label="r ≤ 0 (실패)")
    ax2.legend(handles=[green_p, red_p], fontsize=8)

    # (c) 시너지 강도 vs 상호작용 예측 정확도 산점도
    ax3 = axes[2]
    sc3 = ax3.scatter(syn_mag, r_int, c=syn_ratio, cmap="RdBu_r", vmin=0, vmax=1,
                      s=100, edgecolors="gray", linewidths=0.5)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("실제 상호작용 강도 |Δ(A+B) - ΔA - ΔB|", fontsize=10)
    ax3.set_ylabel("상호작용 예측 Pearson", fontsize=10)
    ax3.set_title("(c) 상호작용 강도 vs 예측 정확도\n(강한 상호작용일수록 예측이 어려운가?)", fontsize=10)
    plt.colorbar(sc3, ax=ax3, label="시너지 비율 (빨강=시너지, 파랑=길항)")
    for i, c in enumerate(conds):
        ax3.annotate(c, (syn_mag[i], r_int[i]), fontsize=5.5, ha="center", va="bottom", alpha=0.8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "interaction_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"그림 저장: {fig_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("Norman 이중 섭동 상호작용 분석")
    log("시너지 vs 길항 + GEARS+HCE 상호작용 예측 평가")
    log("=" * 60)

    log("\n[1] 데이터 로딩...")
    pert_data = load_data()

    log("\n[2] 모델 로딩...")
    gears = load_model(pert_data)

    log("\n[3] 실제 발현 캐시 구축...")
    actual_cache = build_actual_delta_cache(pert_data)

    log("\n[4] 모델 예측 캐시 구축...")
    pred_cache = build_pred_cache(gears, pert_data)

    log("\n[5] 상호작용 분석...")
    results = analyze(gears, pert_data, actual_cache, pred_cache)

    # ── 집계 통계 ──
    log("\n" + "=" * 60)
    log("[ 전체 집계 결과 ]")
    log("=" * 60)

    for split_name in ["combo_seen2", "combo_seen1", "combo_seen0"]:
        entries = results.get(split_name, [])
        if not entries:
            continue
        pearsons = [e["pearson_combo"] for e in entries]
        log(f"\n{split_name} (n={len(entries)})")
        log(f"  Mean Pearson(combo):         {np.mean(pearsons):.4f} ± {np.std(pearsons):.4f}")

        w = [e for e in entries if "pearson_additive_baseline" in e]
        if w:
            add_p  = [e["pearson_additive_baseline"] for e in w]
            log(f"  Mean Pearson(단순합산):      {np.mean(add_p):.4f} ± {np.std(add_p):.4f}")
            model_wins = sum(1 for e in w if e["pearson_combo"] > e["pearson_additive_baseline"])
            log(f"  Model > Additive:            {model_wins}/{len(w)} 케이스")

        w2 = [e for e in entries if "pearson_interaction" in e]
        if w2:
            int_p  = [e["pearson_interaction"] for e in w2]
            syn_r  = [e["actual_synergy_ratio"] for e in w2]
            log(f"  Mean Pearson(상호작용 예측): {np.mean(int_p):.4f} ± {np.std(int_p):.4f}")
            log(f"  Mean 시너지 비율:            {np.mean(syn_r):.3f} "
                f"({'시너지 우세' if np.mean(syn_r) > 0.5 else '길항 우세'})")

    # Top/Bottom 케이스
    all_int = [e for es in results.values() for e in es if "pearson_interaction" in e]
    if all_int:
        all_int.sort(key=lambda x: x["pearson_interaction"], reverse=True)
        log("\n[ 상호작용 예측 Best 5 ]")
        for e in all_int[:5]:
            log(f"  {e['condition']:<30} r={e['pearson_interaction']:.4f}  "
                f"syn_ratio={e.get('actual_synergy_ratio',0):.2f}")
        log("\n[ 상호작용 예측 Worst 5 ]")
        for e in all_int[-5:]:
            log(f"  {e['condition']:<30} r={e['pearson_interaction']:.4f}  "
                f"syn_ratio={e.get('actual_synergy_ratio',0):.2f}")

    all_mag = [e for es in results.values() for e in es if "actual_synergy_magnitude" in e]
    if all_mag:
        all_mag.sort(key=lambda x: x["actual_synergy_magnitude"], reverse=True)
        log("\n[ 상호작용 강도 Top 5 ]")
        for e in all_mag[:5]:
            d = "시너지" if e.get("actual_synergy_ratio", 0) > 0.5 else "길항"
            log(f"  {e['condition']:<30} magnitude={e['actual_synergy_magnitude']:.4f}  ({d})")

    # ── 시각화 ──
    log("\n[6] 시각화...")
    visualize(results)

    # ── 저장 ──
    def convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError

    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    log(f"\n결과 저장: {JSON_PATH}")


if __name__ == "__main__":
    main()
