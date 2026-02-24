# HCE Jacobian Analysis: Ontology-Guided Gene Attribution

> scGPT_brain + HCE GO head를 결합해 **GO 경로별 유전자 기여도**를 Jacobian으로 계산.
> 기존 세포 유형 분류 기반 Jacobian에서 **GO 계층 구조 기반 attribution**으로 확장.

---

## 배경

### 기존 Jacobian 분석의 한계

```
scFoundation → cell_emb → cell_type_logit
                               ↓ Jacobian
               ∂(RG) / ∂(PAX6)   ← "PAX6가 RG 분류에 기여하는가"

문제: 세포 유형이라는 단 하나의 타겟만 존재
     생물학적 경로 수준 해석 불가
     계층 정보 없음 (RG ≠ Neuron ≠ Neural Cell 구별 없음)
```

### HCE Jacobian이 추가하는 것

```
scGPT_brain → cell_emb → HCE GO head → GO_logit_k  (k = 계층 내 모든 노드)
                                             ↓ Jacobian
              ∂(P(apoptosis))  / ∂(CASP3)    ← 리프 수준
              ∂(P(cell_death)) / ∂(CASP3)    ← 중간 수준 (더 크야야 함)
              ∂(P(Neuron))     / ∂(DCX)      ← 세포 유형 계층
              ∂(P(Neural Cell))/ ∂(DCX)      ← 더 상위 계층

→ 같은 유전자에 대해 온톨로지 레벨별 attribution 비교
→ 계층 일관성: ∂(parent) ≥ ∂(child) 검증
→ 경로 특이적 마커 발굴
```

---

## 모델: scGPT_brain

| 항목 | 값 |
|---|---|
| 학습 데이터 | CellxGene Census brain — **13.2M 뇌 세포** |
| d_model | 512 |
| layers | 12 Transformer layers |
| heads | 8 attention heads |
| input style | **continuous** (linear value_encoder) |
| max_seq_len | 1,200 genes |
| n_bins | 51 |

**Gradient 경로 (scFoundation 대비 단순)**:
```
gene_expr (float)
    ↓ value_encoder: Linear(1 → 512)   ← continuous, gradient 직통
    ↓ Transformer × 12
    ↓ cell_emb (mean pooling, 512D)
    ↓ HCE GO head (512 → n_go)
    ↓ GO_logit_k
    ↓ ∂(GO_k) / ∂(gene_expr_i)        ← 목표
```
scFoundation처럼 `DifferentiableTokenEmb` 수동 수정 불필요.

---

## 데이터

| 항목 | 내용 |
|---|---|
| 세포 수 | 116,529 cells |
| 유전자 수 | 49,133 genes |
| 세포 유형 | RG (29k) / Neuroblast (42k) / Ext (27k) / Inh (17k) |
| 발달 단계 | Fetal 1st/2nd/3rd trimester → Neonatal |
| 시간 정보 | `age_days_repr` (49–451일) |

경로 설정은 `HCE_BRAIN_ATLAS` 환경 변수 또는 `config.py`의 `BRAIN_ATLAS`를 통해 지정.

---

## 분석 단계 (Steps)

### Step 1: scGPT_brain + HCE 파인튜닝 (`step1_finetune_hce.py`)

scGPT_brain 위에 HCE Cell Type Classifier를 붙여서 뇌 세포 유형 분류를 학습.

**Brain Cell Ontology (Cell 계층)**:
```
Cell
└── Neural Cell
    ├── Neuron
    │   ├── Excitatory Neuron    → Ext (TBR1, SATB2, SLC17A7)
    │   └── Inhibitory Neuron    → Inh (GAD1, GAD2, DLX2)
    └── Neural Progenitor
        ├── Radial Glia          → RG (PAX6, SOX2, HES1)
        └── Neuroblast           → Neuroblast (DCX, TUBB3, NEUROD1)
```

```python
# 핵심 구조
scgpt_brain = load_pretrained_scgpt(cfg.SCGPT_BRAIN_DIR)
hce_model = ScGPTBrainHCE(
    scgpt_brain,
    n_classes=4,          # RG / Neuroblast / Ext / Inh
    d_model=512,
)
# HierarchicalCrossEntropyLoss로 파인튜닝
# scGPT frozen, HCE head만 학습
```

**출력**: 파인튜닝된 `hce_brain_best.pt` → `jacobian/results/`

### Step 2: HCE Jacobian 계산 (`step2_hce_jacobian.py`)

각 세포에 대해 GO 계층 레벨별 Jacobian 행렬 계산.

```python
# 입력: (B, n_genes) 발현량 텐서, requires_grad=True
x = expr_tensor.requires_grad_(True)

# Forward
output_dict, go_logits = hce_model(gene_ids, x)
all_probs = hce_loss.propagate_probs(torch.sigmoid(go_logits))
# all_probs: (B, n_all_terms)  ← 리프 + 중간 + 루트 모든 노드 확률

# Jacobian: ∂(GO_k) / ∂(x_i)  for each term k
J = torch.autograd.grad(
    all_probs[:, term_idx].sum(), x,
    retain_graph=True
)[0]   # (B, n_genes)
```

**저장**: `results/jacobian_{term}.npy` — 각 GO term별 (n_cells × n_genes) 행렬

### Step 3: 시각화 (`step3_visualize.py`)

1. **레벨별 Top 유전자 비교**
   - `∂(RG) / ∂(gene)` vs `∂(Neuron) / ∂(gene)` vs `∂(Neural Cell) / ∂(gene)`
   - 리프 특이적 마커 vs 범용 신경 마커 구분

2. **계층 일관성 (Monotonicity) 검증**
   - `|∂(parent)| ≥ |∂(child)|` 비율 계산
   - HCE 학습이 Jacobian 레벨에서도 계층 구조를 반영하는지 검증

3. **발달 시계열 분석**
   - `age_days_repr`로 세포를 시간 축 정렬
   - RG → Neuroblast → Ext 분화 경로에서 GO Jacobian 변화 추적
   - 예: `∂(P(Radial Glia)) / ∂(SOX2)` 는 초기 발달에서 높고 후기에서 낮아야 함

4. **GO pathway heatmap**
   - x축: 유전자 (known markers 하이라이트)
   - y축: GO term (계층 순서)
   - 값: mean |Jacobian| per cell type

---

## 기존 분석과 비교

| 분석 | 모델 | 타겟 | 신규 기여 |
|---|---|---|---|
| path_b (기존) | scFoundation | ∂(cell_emb) / ∂(gene) | Expression-level gradient |
| path_d (기존) | scFoundation | Integrated Gradients | Attribution baseline |
| **Step 1-3 (신규)** | **scGPT_brain** | **∂(GO_k) / ∂(gene)** | **Ontology-guided attribution** |

---

## 환경 설정

```bash
# 필요 패키지
pip install scgpt scanpy torch

# 경로 설정 (환경변수로 오버라이드 가능)
export HCE_SCGPT_BRAIN=/path/to/scGPT_brain      # 모델 체크포인트
export HCE_BRAIN_ATLAS=/path/to/brain_atlas.h5ad  # fetal brain 데이터
export HCE_JACOBIAN_RESULTS=/path/to/results       # 결과 저장 경로

# 실행
python -m HCE.jacobian.step1_finetune_hce
python -m HCE.jacobian.step2_hce_jacobian
python -m HCE.jacobian.step3_visualize
```

| 패키지 | 버전 |
|---|---|
| scgpt | 0.2.1 |
| PyTorch | 2.1.2+cu121 |
| Python | 3.10 |
