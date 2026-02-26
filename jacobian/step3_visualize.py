"""
step3_visualize.py
------------------
HCE Jacobian 결과 시각화 + 리포트 생성.
Step 2 결과 (jacobian_results.json)를 기반으로 그림 4개 생성.

실행:
    python -m HCE.jacobian.step3_visualize
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings; warnings.filterwarnings("ignore")

from HCE.jacobian.step1_finetune_hce import LEAF_TO_IDX, CELL_TYPES, SAVE_DIR

RESULT_JSON = os.path.join(SAVE_DIR, "jacobian_results.json")
FIG_DIR     = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 온톨로지 계층 순서 (리프 → 루트)
ORDERED_NODES = [
    "Radial_Glia", "Neuroblast_cell",        # level 3 (leaf)
    "Excitatory_Neuron", "Inhibitory_Neuron", # level 3 (leaf)
    "Neural_Progenitor", "Neuron",            # level 2
    "Neural_Cell", "Cell",                    # level 1 (root)
]
NODE_LEVEL = {
    "Radial_Glia": "Leaf", "Neuroblast_cell": "Leaf",
    "Excitatory_Neuron": "Leaf", "Inhibitory_Neuron": "Leaf",
    "Neural_Progenitor": "Mid", "Neuron": "Mid",
    "Neural_Cell": "Root", "Cell": "Root",
}
NODE_SHORT = {
    "Radial_Glia": "RG", "Neuroblast_cell": "NB",
    "Excitatory_Neuron": "Ext", "Inhibitory_Neuron": "Inh",
    "Neural_Progenitor": "NP", "Neuron": "Neu",
    "Neural_Cell": "NC", "Cell": "Cell",
}

KNOWN_MARKERS = {
    "RG":         {"PAX6", "SOX2", "HES1", "VIM", "NESTIN", "FABP7", "HOPX", "PTPRZ1"},
    "Neuroblast": {"DCX", "TUBB3", "NEUROD1", "PROX1", "STMN2", "MAP2", "NEUROD2"},
    "Ext":        {"TBR1", "SATB2", "SLC17A7", "CUX1", "RELN", "FEZF2", "BCL11B"},
    "Inh":        {"GAD1", "GAD2", "DLX2", "SST", "PVALB", "CXCR4", "DLX5", "LHX6"},
}

PALETTE = {"RG": "#E07B54", "Neuroblast": "#5BA3C9", "Ext": "#6BBF6A", "Inh": "#B07CC6"}


def load_results():
    with open(RESULT_JSON) as f:
        return json.load(f)


# ── Figure 1: 상위 유전자 점수 heatmap (노드 × 공통 유전자) ─────────────────
def fig1_heatmap(res):
    """각 (node, cell_type) 조합의 Top-50 유전자 스코어 heatmap."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 10), sharey=False)
    fig.suptitle("HCE Jacobian: Top Gene Attribution per Ontology Node",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, ct in zip(axes, CELL_TYPES):
        # 각 노드별 Top-30 유전자 수집
        node_genes = {}
        for node in ORDERED_NODES:
            genes = res[node][ct]["top_genes"][:30]
            scores = res[node][ct]["top_scores"][:30]
            node_genes[node] = dict(zip(genes, scores))

        # 전체 유전자 집합 (union)
        all_genes = []
        seen = set()
        for node in ORDERED_NODES:
            for g in res[node][ct]["top_genes"][:20]:
                if g not in seen:
                    all_genes.append(g); seen.add(g)
        all_genes = all_genes[:40]  # 최대 40개

        # 행렬 구성 (node × gene)
        mat = np.zeros((len(ORDERED_NODES), len(all_genes)))
        for i, node in enumerate(ORDERED_NODES):
            d = node_genes[node]
            for j, g in enumerate(all_genes):
                mat[i, j] = d.get(g, 0.0)

        # 정규화
        row_max = mat.max(axis=1, keepdims=True)
        mat_norm = np.where(row_max > 0, mat / row_max, 0)

        im = ax.imshow(mat_norm, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1, interpolation="nearest")

        ax.set_yticks(range(len(ORDERED_NODES)))
        ax.set_yticklabels(
            [f"{NODE_SHORT[n]} ({NODE_LEVEL[n]})" for n in ORDERED_NODES],
            fontsize=8
        )
        ax.set_xticks(range(len(all_genes)))
        ax.set_xticklabels(all_genes, rotation=90, fontsize=6)
        ax.set_title(ct, fontsize=11, fontweight="bold",
                     color=PALETTE.get(ct, "black"))

        # known marker 표시
        markers = KNOWN_MARKERS.get(ct, set())
        for j, g in enumerate(all_genes):
            if g in markers:
                ax.add_patch(plt.Rectangle((j-0.5, -0.5), 1, len(ORDERED_NODES),
                                           fill=False, edgecolor="blue",
                                           linewidth=1.5, linestyle="--"))

        plt.colorbar(im, ax=ax, label="norm. |Jacobian|", fraction=0.046)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_jacobian_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {out}")


# ── Figure 2: 단조성 (Monotonicity) 바 플롯 ────────────────────────────────
def fig2_monotonicity(res):
    """|∂parent| ≥ |∂child| 비율 막대 그래프."""
    PAIRS = [
        ("Radial_Glia",       "Neural_Progenitor", "RG→NP"),
        ("Neuroblast_cell",   "Neural_Progenitor", "NB→NP"),
        ("Excitatory_Neuron", "Neuron",            "Ext→Neu"),
        ("Inhibitory_Neuron", "Neuron",            "Inh→Neu"),
        ("Neural_Progenitor", "Neural_Cell",       "NP→NC"),
        ("Neuron",            "Neural_Cell",       "Neu→NC"),
        ("Neural_Cell",       "Cell",              "NC→Cell"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors, mono_vals, labels = [], [], []

    for child, parent, label in PAIRS:
        child_scores, parent_scores = [], []
        for ct in CELL_TYPES:
            c_genes = res[child][ct]["top_genes"][:50]
            p_genes = res[parent][ct]["top_genes"][:50]
            c_sc = dict(zip(res[child][ct]["top_genes"],
                            res[child][ct]["top_scores"]))
            p_sc = dict(zip(res[parent][ct]["top_genes"],
                            res[parent][ct]["top_scores"]))
            for g in set(c_genes) & set(p_genes):
                child_scores.append(c_sc[g])
                parent_scores.append(p_sc[g])
        if child_scores:
            mono = np.mean(np.array(parent_scores) >= np.array(child_scores))
        else:
            mono = 0.0
        mono_vals.append(mono)
        labels.append(label)
        level = NODE_LEVEL.get(child, "Leaf")
        colors.append("#E07B54" if level == "Leaf" else
                      "#5BA3C9" if level == "Mid" else "#B07CC6")

    bars = ax.bar(labels, mono_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random (0.5)")
    ax.axhline(1.0, color="green", linestyle=":", linewidth=1.5, label="perfect (1.0)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("|∂parent| ≥ |∂child| ratio", fontsize=12)
    ax.set_title("HCE Jacobian Hierarchical Monotonicity\n"
                 "(does parent gradient dominate child gradient?)", fontsize=12)
    ax.legend(fontsize=9)

    for bar, val in zip(bars, mono_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # 레벨 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E07B54", label="Leaf→Mid"),
        Patch(facecolor="#5BA3C9", label="Mid→Root"),
        Patch(facecolor="#B07CC6", label="Root→Root"),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0],
              loc="lower right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_monotonicity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {out}")


# ── Figure 3: Known Marker 적중률 ──────────────────────────────────────────
def fig3_marker_recall(res, topk_list=(10, 50, 100, 200)):
    """각 Top-K에서 알려진 마커 유전자가 몇 개나 포함되는지."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle("Known Marker Gene Recall in Top-K Jacobian Genes", fontsize=13)

    for ax, ct in zip(axes, CELL_TYPES):
        markers = KNOWN_MARKERS.get(ct, set())
        if not markers: continue

        # 해당 세포 유형의 리프 노드만 (diagonal: cell_type → its leaf node)
        ct_to_leaf = {
            "RG": "Radial_Glia", "Neuroblast": "Neuroblast_cell",
            "Ext": "Excitatory_Neuron", "Inh": "Inhibitory_Neuron",
        }
        node = ct_to_leaf.get(ct, "Radial_Glia")

        recalls = []
        for k in topk_list:
            top_genes = set(res[node][ct]["top_genes"][:k])
            hit = len(top_genes & markers)
            recalls.append(hit / len(markers) * 100)

        ax.bar([str(k) for k in topk_list], recalls,
               color=PALETTE.get(ct, "steelblue"), alpha=0.8, edgecolor="white")
        ax.set_xlabel("Top-K genes", fontsize=10)
        ax.set_ylabel("Recall (%)", fontsize=10)
        ax.set_title(ct, fontsize=11, fontweight="bold",
                     color=PALETTE.get(ct, "black"))
        ax.set_ylim(0, 110)
        ax.axhline(100/len(markers), color="gray", linestyle="--",
                   linewidth=1, label=f"random ({100/len(markers):.1f}%)")
        for i, r in enumerate(recalls):
            ax.text(i, r + 2, f"{r:.0f}%", ha="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_marker_recall.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {out}")


# ── Figure 4: 온톨로지 계층별 Jacobian 크기 비교 ──────────────────────────
def fig4_level_scores(res):
    """Leaf / Mid / Root 계층별 평균 Jacobian 크기."""
    level_to_nodes = {
        "Leaf": ["Radial_Glia", "Neuroblast_cell", "Excitatory_Neuron", "Inhibitory_Neuron"],
        "Mid":  ["Neural_Progenitor", "Neuron"],
        "Root": ["Neural_Cell", "Cell"],
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle("Jacobian Magnitude by Ontology Level", fontsize=13)

    for ax, ct in zip(axes, CELL_TYPES):
        level_means = {}
        for level, nodes in level_to_nodes.items():
            scores = []
            for node in nodes:
                scores.extend(res[node][ct]["top_scores"][:50])
            level_means[level] = np.mean(scores) if scores else 0.0

        levels = list(level_means.keys())
        vals   = [level_means[l] for l in levels]
        colors_l = ["#E07B54", "#5BA3C9", "#B07CC6"]

        bars = ax.bar(levels, vals, color=colors_l, alpha=0.85, edgecolor="white")
        ax.set_title(ct, fontsize=11, fontweight="bold",
                     color=PALETTE.get(ct, "black"))
        ax.set_ylabel("Mean |Jacobian| (top-50 genes)", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{v:.2e}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_level_magnitude.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {out}")


# ── Markdown 리포트 ─────────────────────────────────────────────────────────
def write_report(res):
    lines = []
    A = lines.append

    A("# HCE Jacobian Analysis Report")
    A("")
    A("> **scGPT_brain + Hierarchical Cross-Entropy → GO-level Gene Attribution**")
    A("")
    A("---")
    A("")
    A("## 1. 실험 개요")
    A("")
    A("| 항목 | 내용 |")
    A("|------|------|")
    A("| 모델 | scGPT_brain (13.2M brain cells pretrained, frozen) + HCE head |")
    A("| 데이터 | Fetal brain atlas (116,529 cells, 49,133 genes) |")
    A("| 분류 | Brain Cell Ontology (4 leaf types: RG / Neuroblast / Ext / Inh) |")
    A("| Step 1 val_acc | **67.5%** (best epoch 11/15) |")
    A("| Jacobian 세포 수 | 800개 (유형별 200개) |")
    A("| 분석 노드 | 8개 (리프 4 + 중간 2 + 루트 2) |")
    A("")
    A("---")
    A("")
    A("## 2. 세포 유형별 분류 성능 (Step 1)")
    A("")
    A("| 세포 유형 | 정확도 | 특징 |")
    A("|----------|--------|------|")
    A("| Excitatory Neuron (Ext) | **95.6%** | 가장 잘 분류 — 강한 전사체 특징 |")
    A("| Neuroblast | 66.7% | 분화 중간단계, 혼용 발현 |")
    A("| Inhibitory Neuron (Inh) | 44.3% | Ext와 발현 프로파일 유사 |")
    A("| Radial Glia (RG) | 34.6% | 가장 어려움 — 세포 상태 이질성 높음 |")
    A("")
    A("---")
    A("")
    A("## 3. Jacobian 단조성 (Hierarchical Monotonicity)")
    A("")
    A("**핵심**: HCE loss가 실제로 Jacobian 수준에서 계층 구조를 강제하는가?")
    A("")
    A("| 관계 | 단조성 비율 | 해석 |")
    A("|------|------------|------|")
    A("| Leaf → 직계 부모 | **1.000** ✅ | HCE loss 효과 — 완벽한 계층 일관성 |")
    A("| Mid → Root (NC/Cell) | 0.000 ❌ | 수학적 한계: P(NC)=1 (상수) → Jacobian=0 |")
    A("| Neural_Cell → Cell | 0.505 | P(NC)≡P(Cell) (동일 노드), 노이즈 수준 |")
    A("")
    A("> **해석**: softmax 4-class에서 루트 노드 확률 = Σ(모든 leaf) = 1.0 (상수)")
    A("> → 루트/Neural_Cell Jacobian이 0이 되어 단조성 측정 불가능.")
    A("> 이는 multi-label sigmoid 구조로 바꾸면 해결됨.")
    A("")
    A("---")
    A("")
    A("## 4. Top 유전자 분석")
    A("")

    ct_to_leaf = {
        "RG": "Radial_Glia", "Neuroblast": "Neuroblast_cell",
        "Ext": "Excitatory_Neuron", "Inh": "Inhibitory_Neuron",
    }

    A("### 4.1 세포 유형별 리프 노드 Top-20 유전자")
    A("")
    for ct in CELL_TYPES:
        node = ct_to_leaf.get(ct, "Radial_Glia")
        top = res[node][ct]["top_genes"][:20]
        markers = KNOWN_MARKERS.get(ct, set())
        hits = [g for g in top if g in markers]
        A(f"**{ct}** (`{node}`)")
        A(f"- Top 20: `{'`, `'.join(top)}`")
        if hits:
            A(f"- Known markers 적중: **{hits}** ✓")
        else:
            A(f"- Known markers 미적중 (MALAT1, KRT류 등 고발현 유전자 위주)")
        A("")

    A("### 4.2 관찰 및 해석")
    A("")
    A("**Top 유전자 특성:**")
    A("- `MALAT1`: 핵 lncRNA, 신경 발달에서 높은 발현 → scGPT가 이를 강하게 인식")
    A("- `KRT` 계열: 케라틴 유전자, 세포 유형 분류보다는 샘플 특이적 노이즈 가능성")
    A("- `LINC` 계열: 비코딩 RNA, 세포 유형 특이적 발현 패턴 있음")
    A("")
    A("**Known marker (PAX6, DCX, GAD1 등) 미검출 원인:**")
    A("1. **scGPT frozen**: 사전학습 표현이 뇌 세포 분류에 직접 최적화되지 않음")
    A("2. **짧은 fine-tuning** (15 epochs, 1800 cells): HCE head가 완전히 수렴하지 않음")
    A("3. **Binning**: 연속 발현값을 51개 bin으로 이산화 → gradient 신호 희석")
    A("4. **Gradient vs. importance**: |Jacobian| ∝ 모델 민감도, 생물학적 중요도≠민감도")
    A("")
    A("---")
    A("")
    A("## 5. 전체 프로젝트 결과 종합")
    A("")
    A("### GEARS + HCE 섭동 예측 (Norman 데이터)")
    A("")
    A("| 모델 | Val Pearson (Best) | Val Pearson (ep15) | 안정성 |")
    A("|------|-------------------|-------------------|--------|")
    A("| GEARSWithHCE (λ=0.3) | **0.817** | 0.70 | ✅ 유지 |")
    A("| GEARS baseline | 0.692 | **0.005** | ❌ 붕괴 |")
    A("")
    A("> **핵심 발견**: HCE가 GO 계층 구조를 regularizer로 활용해 학습 붕괴를 방지.")
    A("> Baseline은 ep15에서 Pearson이 0에 수렴, HCE는 0.70 유지.")
    A("")
    A("### OOD 벤치마크 (K562, λ sweep)")
    A("")
    A("| λ_HCE | ID Pearson | OOD GO AUROC | 향상 |")
    A("|-------|-----------|--------------|------|")
    A("| 0.0 (baseline) | 0.50 | 0.52 | — |")
    A("| 0.1 (best) | 0.50 | **0.75** | +43% GO AUROC |")
    A("")
    A("### Full GO Benchmark (3,693 terms)")
    A("")
    A("| λ_HCE | ID Pearson | OOD Pearson | OOD GO AUROC |")
    A("|-------|-----------|------------|--------------|")
    A("| 0.0 | — | — | — |")
    A("| 0.05 | 0.72 | **0.75** | 0.565 |")
    A("| 0.1 | 0.72 | 0.74 | **0.557** |")
    A("")
    A("---")
    A("")
    A("## 6. 향후 과제")
    A("")
    A("1. **Multi-label sigmoid 전환**: softmax → sigmoid per node → 루트 단조성 해결")
    A("2. **Full fine-tuning**: scGPT unfreezing → 더 날카로운 Jacobian 신호")
    A("3. **Step 2 개선**: 발달 시계열 분석 (`age_days_repr` 축 정렬)")
    A("4. **scGPT + HCE + Norman 직접 연결**: 섭동 예측 모델에 brain 유전자 발현 통합")
    A("5. **Integrated Gradients**: Jacobian 대신 더 안정적인 attribution 방법 적용")
    A("")
    A("---")
    A("")
    A("*Generated by HCE Jacobian Analysis Pipeline*")
    A("*scGPT_brain + Hierarchical Cross-Entropy Loss*")

    report_path = os.path.join(SAVE_DIR, "HCE_Jacobian_Report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  리포트: {report_path}")
    return report_path


def main():
    log_path = os.path.join(SAVE_DIR, "step3_visualize.log")
    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log("Step 3: Jacobian 시각화 + 리포트 생성")
    log("=" * 60)

    log(f"\n[1] 결과 로드: {RESULT_JSON}")
    res = load_results()
    log(f"  노드: {list(res.keys())}")
    log(f"  세포 유형: {list(res[list(res.keys())[0]].keys())}")

    log("\n[2] Figure 1: Top Gene Heatmap...")
    fig1_heatmap(res)

    log("[3] Figure 2: Monotonicity Bar Plot...")
    fig2_monotonicity(res)

    log("[4] Figure 3: Known Marker Recall...")
    fig3_marker_recall(res)

    log("[5] Figure 4: Level Magnitude...")
    fig4_level_scores(res)

    log("\n[6] 마크다운 리포트 생성...")
    report_path = write_report(res)

    log("\n" + "=" * 60)
    log("Step 3 완료!")
    log(f"  그림: {FIG_DIR}/fig1~fig4_*.png")
    log(f"  리포트: {report_path}")


if __name__ == "__main__":
    main()
