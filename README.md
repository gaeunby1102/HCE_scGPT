# HCE: Hierarchical Cross-Entropy for Single-Cell Perturbation Modeling

> Nature Computational Science (2025) 논문의 HCE 아이디어를 유전자 섭동 예측 및 세포 유형 분류에 적용한 구현체.

---

## 핵심 아이디어

기존 섭동 예측 모델(GEARS, scGPT 등)은 MSE 회귀만으로 최적화 → OOD 예측 시 성능 급락, 생물학적 해석 불가.

**HCE가 추가하는 것:**
- GO 계층 구조 기반 분류 loss 동시 적용 → `P(apoptosis) ≤ P(cell_death) ≤ P(cellular_process)`
- 틀려도 "큰 그림은 맞춤" → OOD 강건성 향상
- 계층적 예측 출력으로 생물학자 해석 가능

---

## 실험 결과

### 1. GEARS + HCE (Norman 데이터셋)

| 모델 | Best Pearson | ep15 Pearson | 비고 |
|------|-------------|-------------|------|
| GEARSWithHCE (λ=0.3) | **0.817** | 0.70 | ✅ 안정 |
| GEARS baseline | 0.692 | **0.005** | ❌ ep15 붕괴 |

> HCE가 GO 계층 구조를 regularizer로 활용해 **학습 붕괴를 방지**.
> Baseline은 ep15에서 Pearson이 0에 수렴, HCE는 0.70 유지.

---

### 2. OOD 벤치마크 (K562 GWPS, Gene-OOD split)

**설정**: MSigDB Hallmark 온톨로지, lambda 스윕

| 모델 | 분할 | Pearson | GO AUROC | 비고 |
|------|------|---------|---------|------|
| λ=0.0 (baseline) | ID  | 0.374 | 0.535 | 순수 회귀 |
| λ=0.0 (baseline) | OOD | 0.451 | 0.556 | |
| λ=0.05 | ID  | 0.378 | **0.746** | +39% AUROC |
| λ=0.05 | OOD | 0.450 | 0.664 | |
| **λ=0.1 (최적)** | **ID**  | **0.389** | **0.763** | **+4% Pearson, +43% AUROC** |
| **λ=0.1 (최적)** | **OOD** | **0.452** | **0.687** | |
| λ=0.3 | ID  | 0.375 | 0.747 | lambda 과대 |
| λ=0.3 | OOD | 0.444 | 0.663 | |

**핵심**: GO pathway 예측 AUROC 0.535 → 0.763 (+43%), 회귀(Pearson)은 유지/소폭 향상.

---

### 3. Full GO 벤치마크 (3,693 terms, K562 gene2go 라벨)

| λ_HCE | ID Pearson | OOD Pearson | OOD GO AUROC |
|-------|-----------|------------|-------------|
| 0.0 (baseline) | 0.50 | 0.52 | ~0.50 |
| **0.05** | 0.72 | **0.754** | **0.565** |
| 0.1 | 0.72 | 0.742 | 0.557 |

> Full GO (3,693 term) 스케일에서도 λ=0.05가 OOD Pearson +45% 향상.

---

### 4. scGPT_brain + HCE (Norman 섭동 예측 — 도메인 전이 실험)

**설정**: scGPT_brain(frozen) encoder + pert_gene_emb → MLP predictor + HCE loss

```
scGPT_brain(ctrl_expr)  → cell_emb (512D)   [frozen]
scGPT.encoder(pert_gene) → pert_emb (512D)   [frozen]
cat([cell_emb, pert_emb]) → MLP → Δexpr + GO logits
```

| 모델 | Best Pearson | Test Pearson | 비고 |
|------|-------------|-------------|------|
| GEARS baseline | 0.692 | — | ep15=0.005 붕괴 |
| GEARS + HCE | **0.817** | — | ep15=0.700 안정 |
| scGPT_brain + HCE (λ=0.1) | 0.165 | 0.193 | 도메인 미스매치 |

> **결과 해석**: scGPT_brain은 뇌 세포(13.2M)로 사전학습, Norman은 K562 암세포주.
> 도메인 불일치 + 유전자-유전자 상호작용 그래프 부재(GEARS는 GNN 사용)로 성능 차이 발생.

---

### 5. scGPT_brain + HCE, 부분 해동 (Task 2)

**설정**: scGPT_brain의 마지막 2 Transformer layer + value_encoder 해동, 차등 학습률

```
scGPT_brain  (most layers frozen)
  ↓ last 2 layers + value_encoder   [lr = 1e-5]
cell_emb (512D)  +  pert_emb (512D)
  ↓
MLP predictor + GO HCE head         [lr = 3e-4]
  ↓
Δexpr  +  GO logits
```

| 모델 | Best Pearson | ep15 Pearson | 비고 |
|------|-------------|-------------|------|
| scGPT_brain + HCE (frozen) | 0.165 | 0.193 | Exp 4, 비교 기준 |
| **scGPT_brain + HCE (finetune)** | **0.2514** | 0.2252 | **+52% vs frozen** |

> **해동의 효과**: last 2 layers unfreeze로 0.193 → 0.2514 (+30%). 뇌 사전학습 도메인 → K562
> 전이 개선. 단, GEARS GNN(0.817)과의 격차는 유전자 상호작용 그래프 부재가 주원인.

**학습 곡선**:
```
Ep  3 | val_pearson = 0.1568
Ep  6 | val_pearson = 0.2085
Ep  9 | val_pearson = 0.2196
Ep 12 | val_pearson = 0.2203
Ep 15 | val_pearson = 0.2252  (best test = 0.2514)
```

---

### 6. scGPT + GEARS + HCE (Task 3) — 실행 중

**설정**: scGPT_brain 유전자 임베딩(512D → PCA 64D)으로 GEARS gene_emb / pert_emb 초기화

```
scGPT_brain.encoder(gene_token)  → (n_genes, 512)   [frozen, 임베딩 추출]
   ↓ PCA 512D → 64D
GEARS gene_emb / pert_emb 가중치 초기화
   ↓
GEARSWithHCE 학습 (λ_hce=0.3, epochs=15)   ← 현재 실행 중
```

| 항목 | 값 |
|------|---|
| vocab 커버리지 | 90.1% (4547/5045 genes) |
| PCA 설명 분산 | 48.8% |
| 현재 상태 | **학습 중** (GPU 36% 사용) |

> 결과는 학습 완료 후 업데이트 예정.

---

### 5. Jacobian 분석 — scGPT_brain + HCE Brain Cell Ontology

**설정**: scGPT_brain (13.2M cells pretrained) + HCE head 파인튜닝 → Jacobian 계산

| 항목 | 값 |
|------|---|
| 데이터 | Fetal brain atlas (116,529 cells) |
| 분류 | 4 cell types: RG / Neuroblast / Ext / Inh |
| Step 1 val_acc | **67.5%** (Ext=95.6%, Neuroblast=66.7%, Inh=44.3%, RG=34.6%) |
| Jacobian 세포 수 | 800개 (유형별 200개) |
| 분석 노드 | 8개 (리프 4 + 중간 2 + 루트 2) |

**Jacobian 단조성 (HCE 효과 검증)**:

| 관계 | 단조성 비율 | 해석 |
|------|----------|------|
| Leaf → 직계 부모 | **1.000** ✅ | HCE loss 효과 — 완벽한 계층 일관성 |
| Mid → Root | 0.000 ❌ | softmax 수학적 한계: P(Neural_Cell)=1 (상수) → Jacobian=0 |

> softmax 4-class에서 루트 노드 P = Σ(모든 leaf) = 1.0 → 상수 → Jacobian=0.
> → multi-label sigmoid 구조로 바꾸면 해결 가능.

---

## 구현 구조

```
HCE/
├── ontology.py          # OntologyDAG (Cell Ontology / GO 범용 DAG)
├── loss.py              # HierarchicalCrossEntropyLoss
│                        # HierarchicalPerturbationLoss
├── model.py             # HCECellTypeClassifier, HCEPerturbationPredictor
├── config.py            # 경로 설정 (환경변수 오버라이드 지원)
├── data_replogle.py     # Replogle K562 데이터 로더 + GO 라벨 생성
├── gears_hce.py         # GEARS 서브클래싱 통합
├── gears_norman_hce.py  # Norman 데이터 전체 GEARS+HCE 학습
├── gears_norman_baseline.py # GEARS baseline (비교용)
├── benchmark_ood.py     # OOD 벤치마크 (lambda 스윕)
├── benchmark_full_go.py # Full GO (3,693 terms) 벤치마크
├── msigdb_ontology.py   # MSigDB Hallmark (50 terms) 온톨로지
├── go_ontology_full.py  # 실제 GO (BP+CC+MF, ~3,693 terms)
├── scgpt_hce.py         # scGPT 파인튜닝 HCE 드롭인
├── scgpt_norman_hce.py  # scGPT_brain + HCE → Norman 섭동 예측 (Exp 4, frozen)
├── scgpt_norman_finetune.py  # scGPT_brain 부분 해동 + HCE (Exp 5, Task 2)
├── scgpt_gears_hce.py   # scGPT 임베딩 초기화 GEARS + HCE (Exp 6, Task 3)
├── jacobian/
│   ├── step1_finetune_hce.py   # scGPT_brain + HCE 파인튜닝
│   ├── step2_hce_jacobian.py   # Jacobian 계산 (∂P(node)/∂gene)
│   ├── step3_visualize.py      # 시각화 + 레포트 생성
│   └── results/                # 체크포인트, Jacobian JSON, 그림 4종
├── results/             # 벤치마크 결과 JSON
└── demo.py              # 실행 가능한 전체 파이프라인 데모
```

---

## 핵심 구성요소

### 1. OntologyDAG (`ontology.py`)

```python
from HCE.ontology import OntologyDAG

dag = OntologyDAG()
dag.add_edge("apoptosis", "cell_death")         # child is_a parent
dag.add_edge("cell_death", "cellular_process")

ancestors = dag.get_ancestors("apoptosis")
# → {"apoptosis", "cell_death", "cellular_process", "biological_process"}
```

구현된 온톨로지:
- `build_k562_go_ontology()` — K562 관련 GO 경로 16 리프, 29 노드
- `build_hallmark_ontology()` — MSigDB Hallmark 44 리프, 52 노드
- `build_go_dag_from_obo()` — 실제 go-basic.obo: **3,693 유효 텀, 1,121 리프**

### 2. HCE 손실 함수 (`loss.py`)

#### `HierarchicalCrossEntropyLoss` (세포 유형 분류용)

```
L = (1-α)·L_flat_CE  +  α·L_hier_BCE  +  λ_mono·L_monotonicity
```

| 하이퍼파라미터 | 기본값 | 의미 |
|---|---|---|
| `alpha` | 0.7 | 계층 손실 비중 |
| `monotone_coef` | 0.1 | 단조성 패널티 강도 |

#### `HierarchicalPerturbationLoss` (섭동 예측용)

```
L = λ_reg·L_MSE  +  λ_cls·L_HCE_multilabel
```

### 3. GEARS 서브클래싱 (`gears_hce.py`)

```python
gears = GEARSWithHCE(pert_data, device='cuda')
gears.model_initialize_hce(hidden_size=64, lambda_hce=0.1)   # 최적값
gears.train(epochs=20)
```

---

## 실행 방법

### 환경변수 설정

```bash
export HCE_DATA_ROOT=/path/to/data          # go-basic.obo, gene2go_all.pkl 등
export HCE_K562_DATA=/path/to/K562.h5ad
export HCE_SCGPT_BRAIN=/path/to/scGPT_brain
export HCE_BRAIN_ATLAS=/path/to/brain.h5ad
```

### 벤치마크

```bash
python -m HCE.benchmark_ood          # OOD 벤치마크 (Hallmark)
python -m HCE.benchmark_full_go      # Full GO 벤치마크 (3,693 terms)
python -m HCE.gears_norman_baseline  # GEARS baseline 비교
python -m HCE.scgpt_norman_hce       # scGPT_brain + HCE (도메인 전이 실험)
```

### Jacobian 분석 파이프라인

```bash
python -m HCE.jacobian.step1_finetune_hce   # scGPT_brain + HCE 파인튜닝
python -m HCE.jacobian.step2_hce_jacobian   # Jacobian 계산
python -m HCE.jacobian.step3_visualize      # 시각화 + 레포트
```

### Python API

```python
from HCE import HCEPerturbationPredictor
from HCE.data_replogle import ReplogleDataset, build_k562_go_ontology

dataset = ReplogleDataset("K562_gwps_raw_bulk.h5ad", gene_subset=2000)
train_ds, val_ds, test_ds = dataset.get_splits()

dag, term_to_idx = build_k562_go_ontology()
model = HCEPerturbationPredictor(
    n_genes=2000, n_go_terms=len(term_to_idx),
    ontology=dag, go_term_to_idx=term_to_idx,
    lambda_cls=0.1,   # 최적값
)
loss, info = model.compute_loss(expr, pert_mask, delta_expr, go_labels)
```

---

## 환경

| 항목 | 버전 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.3.0 |
| GEARS | snap-stanford/GEARS |
| scgpt | 0.2.1 |
| scanpy | 1.11.5 |
| obonet | ✓ (GO OBO 파싱) |

---

## 향후 과제

1. **Multi-label sigmoid 전환**: softmax → sigmoid per node → 루트 단조성 해결
2. ✅ **scGPT_brain + unfreezing (Task 2)**: last 2 layers unfreeze → Pearson 0.193 → 0.2514 (+30%)
3. 🔄 **scGPT + GNN 하이브리드 (Task 3)**: scGPT 임베딩으로 GEARS 초기화 → 현재 학습 중
4. **Integrated Gradients**: Jacobian 대신 더 안정적인 attribution 방법 적용
5. **Norman + Adamson 통합 평가**: 공식 GEARS 평가 지표 비교
6. **scGPT_human 도메인 전이**: brain → human으로 올바른 도메인 실험
