"""
deep_interaction_analysis.py
-----------------------------
상호작용 예측이 잘 되는 케이스 심층 분석.

분석 항목:
  1. Gene-level 상호작용 프로파일: 실제 vs 예측 (best/worst 5 pairs)
  2. GO term 공유도: 잘 예측된 쌍 vs 못 예측된 쌍
  3. HCE 그래프 근접도: GO graph 최단 거리
  4. 단일 효과 크기 vs 상호작용 예측 정확도
  5. 쌍별 top-interacting 유전자 목록

실행:
    cd /data2/Atlas_Normal
    conda run -n gears2 python -m HCE.deep_interaction_analysis
"""

from __future__ import annotations
import sys, os, json, pickle
sys.path.insert(0, "/data2/Atlas_Normal")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr
from copy import deepcopy
import networkx as nx

from gears import PertData
from gears.utils import print_sys

from HCE.gears_hce import GEARSWithHCE, GEARSModelWithHCE
from HCE.config import GEARS_DATA_DIR, GEARS_SAVE_NORMAN, RESULTS_ROOT

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(GEARS_SAVE_NORMAN, "model.pt")
LOG_PATH   = os.path.join(RESULTS_ROOT, "deep_interaction_analysis.log")
JSON_PATH  = os.path.join(RESULTS_ROOT, "deep_interaction_analysis.json")
FIG_DIR    = os.path.join(RESULTS_ROOT, "figures")
PREV_JSON  = os.path.join(RESULTS_ROOT, "interaction_analysis.json")

# ─── 분석 대상 top/bottom N ───────────────────────────────────────────────────
N_CASES = 5


def log(msg: str):
    print_sys(msg)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ── 데이터 / 모델 로딩 ────────────────────────────────────────────────────────
def load_data():
    pert_data = PertData(GEARS_DATA_DIR)
    pert_data.load(data_path=os.path.join(GEARS_DATA_DIR, "norman"))
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    return pert_data


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


# ── 발현 캐시 ─────────────────────────────────────────────────────────────────
def build_actual_delta_cache(pert_data) -> dict[str, np.ndarray]:
    adata = pert_data.adata
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)
    ctrl_mask = adata.obs["condition"] == "ctrl"
    ctrl_mean = X[ctrl_mask].mean(axis=0)
    cache = {}
    for cond in adata.obs["condition"].unique():
        mask = adata.obs["condition"] == cond
        cache[cond] = X[mask].mean(axis=0) - ctrl_mean
    log(f"  실제 발현 캐시: {len(cache)} 조건")
    return cache


@torch.no_grad()
def build_pred_cache(gears, pert_data) -> dict[str, np.ndarray]:
    gears.model.eval()
    cache_sum, cache_count = {}, {}
    for loader_key in ["test_loader", "val_loader", "train_loader"]:
        loader = pert_data.dataloader.get(loader_key)
        if loader is None:
            continue
        log(f"  예측 캐시: {loader_key} 처리 중...")
        for batch in loader:
            batch.to(DEVICE)
            pred_out, _ = gears.model(batch)
            n_genes = gears.num_genes
            pred = pred_out.reshape(-1, n_genes).cpu().numpy()
            perts = np.array(batch.pert)
            for p in set(perts):
                idx = np.where(perts == p)[0]
                arr = pred[idx].sum(axis=0)
                cnt = len(idx)
                if p not in cache_sum:
                    cache_sum[p]   = arr
                    cache_count[p] = cnt
                else:
                    cache_sum[p]   += arr
                    cache_count[p] += cnt
    pred_cache = {p: cache_sum[p] / cache_count[p] for p in cache_sum}
    log(f"  예측 캐시: {len(pred_cache)} 조건")
    return pred_cache


# ── GO 그래프 분석 ─────────────────────────────────────────────────────────────
def load_go_graph(gears) -> tuple[nx.DiGraph | None, dict]:
    """HCE의 G_go와 gene→GO term 매핑 로드."""
    config_pkl = os.path.join(GEARS_SAVE_NORMAN, "config.pkl")
    with open(config_pkl, "rb") as f:
        saved_config = pickle.load(f)
    G_go = saved_config.get("G_go")
    gene_list = gears.gene_list  # gene_list (str list)
    return G_go, gene_list


def go_overlap(g1: str, g2: str, gene2go: dict) -> tuple[int, float]:
    """두 유전자의 GO term Jaccard overlap (pert_data.gene2go 사용)."""
    t1 = set(gene2go.get(g1, []))
    t2 = set(gene2go.get(g2, []))
    if not t1 or not t2:
        return 0, 0.0
    inter = len(t1 & t2)
    jaccard = inter / len(t1 | t2)
    return inter, jaccard


# ── 핵심 분석: 유전자별 상호작용 프로파일 ────────────────────────────────────
def compute_interaction_vectors(
    cond: str, actual_cache: dict, pred_cache: dict
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(actual_inter, pred_inter) 유전자 벡터 반환."""
    g1, g2 = cond.split("+")
    all_conds = set(actual_cache.keys())

    single_A = next((c for c in [f"{g1}+ctrl", f"ctrl+{g1}"] if c in all_conds), None)
    single_B = next((c for c in [f"{g2}+ctrl", f"ctrl+{g2}"] if c in all_conds), None)

    if single_A is None or single_B is None:
        return None, None

    actual_AB = actual_cache.get(cond)
    actual_A  = actual_cache.get(single_A)
    actual_B  = actual_cache.get(single_B)
    pred_AB   = pred_cache.get(cond)
    pred_A    = pred_cache.get(single_A)
    pred_B    = pred_cache.get(single_B)

    if any(x is None for x in [actual_AB, actual_A, actual_B, pred_AB, pred_A, pred_B]):
        return None, None

    actual_inter = actual_AB - (actual_A + actual_B)
    pred_inter   = pred_AB   - (pred_A   + pred_B)
    return actual_inter, pred_inter


# ── 시각화 1: gene-level profile (best/worst N) ───────────────────────────────
def plot_interaction_profiles(
    cases_best: list[dict], cases_worst: list[dict],
    actual_cache: dict, pred_cache: dict,
    gene_names: list[str],
):
    """각 pair에 대해 gene-level 실제 vs 예측 상호작용 비교."""
    os.makedirs(FIG_DIR, exist_ok=True)

    all_cases = [("Best", cases_best, "#27ae60"), ("Worst", cases_worst, "#e74c3c")]
    n_cols = N_CASES
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4.5, 9))
    fig.suptitle("유전자별 상호작용 프로파일: 예측 Best/Worst 쌍", fontsize=13)

    for row, (label, cases, color) in enumerate(all_cases):
        for col, entry in enumerate(cases[:n_cols]):
            ax = axes[row, col]
            cond = entry["condition"]
            actual_inter, pred_inter = compute_interaction_vectors(
                cond, actual_cache, pred_cache
            )
            if actual_inter is None:
                ax.set_title(f"{cond}\n(데이터 없음)", fontsize=8)
                continue

            # 실제 상호작용 크기 기준 정렬
            order = np.argsort(np.abs(actual_inter))[::-1]
            top_k = 30
            top_idx = order[:top_k]

            actual_top = actual_inter[top_idx]
            pred_top   = pred_inter[top_idx]
            top_genes  = [gene_names[i] for i in top_idx]

            x = np.arange(top_k)
            ax.bar(x - 0.2, actual_top, 0.4, color="#3498db", alpha=0.8, label="실제")
            ax.bar(x + 0.2, pred_top,   0.4, color=color,     alpha=0.7, label="예측")
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xticks(x[:10])
            ax.set_xticklabels(top_genes[:10], rotation=45, ha="right", fontsize=6)
            r_int = entry.get("pearson_interaction", float("nan"))
            ax.set_title(
                f"{label}: {cond}\nr(상호작용)={r_int:.3f}",
                fontsize=8, color=color
            )
            if col == 0:
                ax.set_ylabel("상호작용 Δ(A+B)-(ΔA+ΔB)", fontsize=8)
            if col == 0 and row == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "deep_interaction_profiles.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"그림 저장: {fig_path}")


# ── 시각화 2: 요인 분석 (GO overlap, synergy magnitude, single-KO magnitude) ──
def plot_factor_analysis(all_entries: list[dict], gene2go: dict):
    """상호작용 예측 정확도에 영향하는 요인 분석."""
    os.makedirs(FIG_DIR, exist_ok=True)

    entries = [e for e in all_entries if "pearson_interaction" in e]
    if not entries:
        log("factor 분석: 데이터 없음")
        return {}

    r_int       = np.array([e["pearson_interaction"] for e in entries])
    syn_mag     = np.array([e["actual_synergy_magnitude"] for e in entries])
    syn_ratio   = np.array([e["actual_synergy_ratio"] for e in entries])
    pred_mag    = np.array([e.get("pred_synergy_magnitude", 0) for e in entries])

    # GO term Jaccard overlap 계산
    go_jaccards = []
    for e in entries:
        _, j = go_overlap(e["g1"], e["g2"], gene2go)
        go_jaccards.append(j)
    go_jaccards = np.array(go_jaccards)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("상호작용 예측 정확도 요인 분석", fontsize=13)

    def scatter_with_corr(ax, x, y, xlabel, ylabel, title, c=None):
        valid = ~(np.isnan(x) | np.isnan(y))
        xv, yv = x[valid], y[valid]
        sc = ax.scatter(xv, yv, c=c[valid] if c is not None else "#3498db",
                        cmap="RdBu_r" if c is not None else None,
                        vmin=0 if c is not None else None,
                        vmax=1 if c is not None else None,
                        s=60, alpha=0.7, edgecolors="gray", linewidths=0.3)
        if c is not None:
            plt.colorbar(sc, ax=ax, label="시너지 비율")
        if len(xv) > 2:
            r, p = pearsonr(xv, yv)
            ax.set_title(f"{title}\nr={r:.3f}, p={p:.3f}", fontsize=10)
        else:
            ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        return r if len(xv) > 2 else float("nan")

    factor_corrs = {}
    factor_corrs["synergy_magnitude"] = scatter_with_corr(
        axes[0, 0], syn_mag, r_int,
        "실제 상호작용 강도 |Δ(A+B)-ΔA-ΔB|", "상호작용 예측 Pearson",
        "(a) 강도 vs 예측 정확도", c=syn_ratio
    )
    factor_corrs["synergy_ratio"] = scatter_with_corr(
        axes[0, 1], syn_ratio, r_int,
        "시너지 비율 (0=길항, 1=시너지)", "상호작용 예측 Pearson",
        "(b) 시너지 비율 vs 예측 정확도"
    )
    factor_corrs["go_jaccard"] = scatter_with_corr(
        axes[0, 2], go_jaccards, r_int,
        "GO term Jaccard 유사도", "상호작용 예측 Pearson",
        "(c) GO 공유도 vs 예측 정확도", c=syn_ratio
    )
    factor_corrs["pred_magnitude_ratio"] = scatter_with_corr(
        axes[1, 0], pred_mag / (syn_mag + 1e-8), r_int,
        "예측 강도 / 실제 강도 비율", "상호작용 예측 Pearson",
        "(d) 강도 추정 배율 vs 예측 정확도"
    )

    # (e) best vs worst 비교 box
    ax5 = axes[1, 1]
    sorted_idx = np.argsort(r_int)
    n_q = max(1, len(r_int) // 5)
    bottom_idx = sorted_idx[:n_q]
    top_idx    = sorted_idx[-n_q:]
    data_go    = [go_jaccards[bottom_idx], go_jaccards[top_idx]]
    bp = ax5.boxplot(data_go, labels=["Worst 20%", "Best 20%"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#e74c3c", "#27ae60"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax5.set_ylabel("GO Jaccard 유사도", fontsize=9)
    ax5.set_title("(e) Best vs Worst: GO 공유도", fontsize=10)

    # (f) scatter: GO jaccard vs 상호작용 강도
    scatter_with_corr(
        axes[1, 2], go_jaccards, syn_mag,
        "GO term Jaccard 유사도", "실제 상호작용 강도",
        "(f) GO 공유도 vs 상호작용 강도", c=syn_ratio
    )

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "deep_factor_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"그림 저장: {fig_path}")

    return factor_corrs


# ── 케이스별 상세 출력 ────────────────────────────────────────────────────────
def print_case_detail(entry: dict, actual_cache: dict, pred_cache: dict,
                      gene_names: list[str], gene2go: dict, label: str):
    cond = entry["condition"]
    g1, g2 = entry["g1"], entry["g2"]
    log(f"\n  [{label}] {cond}  r_int={entry['pearson_interaction']:.4f}")
    log(f"    r(combo)={entry['pearson_combo']:.4f}  r(additive)={entry['pearson_additive_baseline']:.4f}")
    log(f"    실제 시너지 비율={entry['actual_synergy_ratio']:.3f}  강도={entry['actual_synergy_magnitude']:.5f}")

    # GO overlap
    t1 = set(gene2go.get(g1, []))
    t2 = set(gene2go.get(g2, []))
    n_inter, jaccard = go_overlap(g1, g2, gene2go)
    log(f"    GO terms: {g1}={len(t1)}, {g2}={len(t2)}, overlap={n_inter}, Jaccard={jaccard:.3f}")

    # 공유 GO term 이름 (최대 5개)
    shared = list(t1 & t2)[:5]
    if shared:
        log(f"    공유 GO terms: {', '.join(shared)}")

    # top 10 interacting genes
    actual_inter, pred_inter = compute_interaction_vectors(cond, actual_cache, pred_cache)
    if actual_inter is not None:
        order = np.argsort(np.abs(actual_inter))[::-1][:10]
        log(f"    실제 top-10 상호작용 유전자:")
        for i in order:
            direction = "SYN" if actual_inter[i] > 0 else "ANT"
            pred_val  = pred_inter[i] if pred_inter is not None else 0
            log(f"      {gene_names[i]:<12} {direction}  actual={actual_inter[i]:+.4f}  pred={pred_val:+.4f}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("Norman 이중 섭동 심층 상호작용 분석")
    log("=" * 60)

    log("\n[1] 이전 결과 로딩...")
    with open(PREV_JSON) as f:
        prev = json.load(f)
    all_entries = []
    for split in ["combo_seen1", "combo_seen0"]:
        all_entries.extend(prev.get(split, []))
    log(f"  총 {len(all_entries)} 조건")

    # best/worst N 선택
    w = [e for e in all_entries if "pearson_interaction" in e]
    w.sort(key=lambda x: x["pearson_interaction"], reverse=True)
    best_cases  = w[:N_CASES]
    worst_cases = w[-N_CASES:][::-1]

    log("\n[2] 데이터 / 모델 로딩...")
    pert_data  = load_data()
    gears      = load_model(pert_data)
    gene_names = gears.gene_list
    gene2go    = pert_data.gene2go  # {gene: [GO:xxx, ...]}

    log("\n[3] 발현 캐시 구축...")
    actual_cache = build_actual_delta_cache(pert_data)
    pred_cache   = build_pred_cache(gears, pert_data)

    # ── 케이스별 상세 출력 ──
    log("\n" + "=" * 60)
    log(f"[ 상호작용 예측 Best {N_CASES} 케이스 ]")
    log("=" * 60)
    for e in best_cases:
        print_case_detail(e, actual_cache, pred_cache, gene_names, gene2go, "BEST")

    log("\n" + "=" * 60)
    log(f"[ 상호작용 예측 Worst {N_CASES} 케이스 ]")
    log("=" * 60)
    for e in worst_cases:
        print_case_detail(e, actual_cache, pred_cache, gene_names, gene2go, "WORST")

    # ── 시각화 ──
    log("\n[4] 유전자별 프로파일 시각화...")
    plot_interaction_profiles(best_cases, worst_cases, actual_cache, pred_cache, gene_names)

    log("\n[5] 요인 분석 시각화...")
    factor_corrs = plot_factor_analysis(w, gene2go)

    log("\n[ 요인별 상관 요약 ]")
    log(f"  상호작용 강도 vs 예측 정확도:   r={factor_corrs.get('synergy_magnitude', 0):.3f}")
    log(f"  시너지 비율  vs 예측 정확도:    r={factor_corrs.get('synergy_ratio', 0):.3f}")
    log(f"  GO Jaccard   vs 예측 정확도:    r={factor_corrs.get('go_jaccard', 0):.3f}")

    # ── 저장 ──
    def convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError

    out = {
        "best_cases":    best_cases,
        "worst_cases":   worst_cases,
        "factor_corrs":  {k: float(v) for k, v in factor_corrs.items()},
    }
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2, default=convert)
    log(f"\n결과 저장: {JSON_PATH}")


if __name__ == "__main__":
    main()
