# HCE Jacobian Analysis: Ontology-Guided Gene Attribution

> scGPT_brain + HCE Brain Cell Ontology 파인튜닝 → **∂P(ontology_node) / ∂gene_expression** Jacobian 계산.
> GO 계층 구조가 Jacobian 수준에서도 단조성을 강제하는지 실험적으로 검증.

---

## 실험 결과 요약

| 항목 | 결과 |
|------|------|
| Step 1 val_acc | **67.5%** (best epoch 11/15) |
| Ext (Excitatory Neuron) | **95.6%** |
| Neuroblast | 66.7% |
| Inh (Inhibitory Neuron) | 44.3% |
| RG (Radial Glia) | 34.6% |
| Jacobian 단조성 (leaf→parent) | **1.000** ✅ |
| Jacobian 단조성 (mid→root) | 0.000 ❌ (softmax 수학적 한계) |

---

## 모델 구조

```
gene_values (float, requires_grad=True)
    ↓ scGPT_brain value_encoder: Linear(1 → 512)   ← continuous, gradient 직통
    ↓ Transformer × 12 layers (frozen)
    ↓ cell_emb (B, 512) — mean pooling
    ↓ HCE head: LayerNorm → Linear(512→256) → GELU → Dropout → Linear(256→4)
    ↓ logits (B, 4)  →  softmax  →  leaf_probs
    ↓ node_prob(node) = Σ leaf_probs[descendants]
    ↓ ∂P(node) / ∂gene_values    ← Jacobian 목표
```

**scGPT_brain**: CellxGene Census 13.2M 뇌 세포 사전학습 (d_model=512, 12 layers, 8 heads)
scGPT는 frozen, HCE head만 학습 (15 epochs, 1800 cells).

---

## Brain Cell Ontology

```
Cell
└── Neural Cell
    ├── Neuron
    │   ├── Excitatory Neuron  → Ext  (TBR1, SATB2, SLC17A7)
    │   └── Inhibitory Neuron  → Inh  (GAD1, GAD2, DLX2)
    └── Neural Progenitor
        ├── Radial Glia        → RG   (PAX6, SOX2, HES1)
        └── Neuroblast         → Neuroblast (DCX, TUBB3, NEUROD1)
```

8개 노드 분석: 리프 4개 + 중간 2개 (Neuron, Neural_Progenitor) + 루트 2개 (Neural_Cell, Cell)

---

## Jacobian 단조성 (핵심 결과)

HCE loss가 Jacobian 수준에서 계층 구조를 강제하는가?

| child → parent | 단조성 비율 | 해석 |
|----------------|-----------|------|
| Radial_Glia → Neural_Progenitor | **1.000** ✅ | 완벽한 계층 일관성 |
| Neuroblast_cell → Neural_Progenitor | **1.000** ✅ | |
| Excitatory_Neuron → Neuron | **1.000** ✅ | |
| Inhibitory_Neuron → Neuron | **1.000** ✅ | |
| Neural_Progenitor → Neural_Cell | 0.000 ❌ | softmax 한계 |
| Neuron → Neural_Cell | 0.000 ❌ | softmax 한계 |
| Neural_Cell → Cell | 0.505 | P(NC)≡P(Cell), 노이즈 수준 |

**softmax 수학적 한계**: 4-class softmax에서 P(Neural_Cell) = Σ(모든 leaf) = 1.0 (상수) → Jacobian = 0.
→ **multi-label sigmoid 구조로 전환하면 해결** 가능.

---

## Top 유전자 분석

### 세포 유형별 리프 노드 Top-20

| 세포 유형 | Top-5 유전자 | Known marker 적중 |
|----------|-------------|-----------------|
| RG | MALAT1, KRT36, KRT32, KLRG2, KLHL7 | ❌ |
| Neuroblast | MALAT1, KRT36, KCNMA1-AS3, KRT1, KRT18P61 | ❌ |
| Ext | KRT36, KRT32, LAIR2, LINC01448, KRT28 | ❌ |
| Inh | MALAT1, KRT36, SOX2-OT, KC6, KRT32 | ❌ (SOX2-OT 근접) |

**Top 유전자 패턴:**
- `MALAT1`: 핵 lncRNA, 신경 발달 전반에서 고발현 → scGPT 강하게 인식
- `KRT` 계열: 케라틴 유전자, 세포 유형보다 샘플 특이적 노이즈 가능성
- `LINC` 계열: 비코딩 RNA, 세포 유형 특이적 패턴 있음

**Known marker (PAX6, DCX, GAD1 등) 미검출 원인:**
1. scGPT frozen → 뇌 세포 분류에 직접 최적화되지 않은 표현
2. 짧은 fine-tuning (15 epochs, 1800 cells) → HCE head 미수렴
3. 51-bin 이산화 → gradient 신호 희석
4. |Jacobian| ∝ 모델 민감도 ≠ 생물학적 중요도

---

## 분석 파이프라인

### Step 1: scGPT_brain + HCE 파인튜닝 (`step1_finetune_hce.py`)

```bash
python -m HCE.jacobian.step1_finetune_hce
```

출력: `results/hce_brain_best.pt` (val_acc=67.5%)

### Step 2: Jacobian 계산 (`step2_hce_jacobian.py`)

```bash
python -m HCE.jacobian.step2_hce_jacobian
```

800개 세포 (유형별 200개) × 8 온톨로지 노드 × 1200 유전자 위치
출력: `results/jacobian_results.json` (노드별 × 세포유형별 Top-200 유전자)

```python
# Jacobian 계산 핵심
vals = values.to(DEVICE).float().requires_grad_(True)
logits, _ = model(gene_ids, vals, pad_mask)
leaf_probs = torch.softmax(logits, dim=-1)       # (B, 4)
prob = leaf_probs[:, node_leaf_indices].sum(-1)  # (B,) — 노드 확률
g = torch.autograd.grad(prob.sum(), vals,
                        retain_graph=True)[0]     # (B, L) — Jacobian
```

### Step 3: 시각화 (`step3_visualize.py`)

```bash
python -m HCE.jacobian.step3_visualize
```

생성 파일:
| 파일 | 내용 |
|------|------|
| `figures/fig1_jacobian_heatmap.png` | 노드 × 세포유형 Top gene heatmap |
| `figures/fig2_monotonicity.png` | child→parent 단조성 비율 bar chart |
| `figures/fig3_marker_recall.png` | Known marker Top-K recall |
| `figures/fig4_level_magnitude.png` | 온톨로지 레벨별 |Jacobian| 크기 |
| `HCE_Jacobian_Report.md` | 전체 분석 레포트 |

---

## 환경 설정

```bash
export HCE_SCGPT_BRAIN=/path/to/scGPT_brain      # 모델 체크포인트 디렉토리
export HCE_BRAIN_ATLAS=/path/to/brain_atlas.h5ad  # fetal brain 데이터
export HCE_JACOBIAN_RESULTS=/path/to/results       # 결과 저장 경로
```

| 패키지 | 버전 |
|--------|------|
| scgpt | 0.2.1 |
| PyTorch | 2.1.2+ |
| Python | 3.10+ |
| scanpy | 1.11.5 |

---

## 향후 과제

1. **Multi-label sigmoid 전환**: softmax → sigmoid per node → 루트 단조성 해결
2. **Full fine-tuning**: scGPT unfreezing → 더 날카로운 Jacobian 신호
3. **Integrated Gradients**: Jacobian 대신 더 안정적인 attribution 방법 적용
4. **발달 시계열 분석**: `age_days_repr` 축 정렬 → RG→Neuroblast→Ext 분화 경로 추적
5. **scGPT + HCE + Norman 연결**: 섭동 예측에 brain 유전자 발현 통합
