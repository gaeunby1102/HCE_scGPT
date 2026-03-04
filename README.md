# HCE: Hierarchical Cross-Entropy for Single-Cell Perturbation Prediction

> **한 줄 요약**: 유전자 섭동(CRISPR KO) 실험 결과를 예측하는 AI 모델에 GO(Gene Ontology) 계층 구조를 손실 함수로 추가해 학습 안정성과 생물학적 해석 가능성을 동시에 높인 구현체.

---

## 배경: 어떤 문제를 풀고 있나?

### 단일세포 유전자 섭동 예측

CRISPR 기술로 특정 유전자를 KO(knock-out)하면 세포 내 수천 개 유전자의 발현량이 연쇄적으로 변한다. 이 변화 패턴을 **사전에 예측**할 수 있다면, 방대한 실험 없이 약물 타겟을 찾거나 세포 운명을 조작하는 법을 알 수 있다.

```
유전자 A KO → 세포 내 5,045 유전자 발현량 변화 예측
               (실험 없이 in silico로)
```

### 기존 방법의 한계: 학습 붕괴

[GEARS](https://www.nature.com/articles/s41587-023-01905-6) 같은 최신 GNN 기반 모델도 훈련 후반부에 성능이 급락하는 **학습 붕괴** 현상이 나타난다.

```
GEARS baseline — ep1: Pearson 0.69 → ep15: Pearson 0.005  ← 붕괴!
```

### HCE가 해결하는 것

기존 **MSE 회귀 손실**에 **GO 계층 분류 손실**을 추가해 붕괴를 방지한다.

```
GO 계층 구조 (Gene Ontology):

biological_process
  └── cellular_process
        ├── cell_cycle        ← 섭동 유전자가 이 경로에 속한다면
        └── cell_death             모든 조상 노드도 "양성"으로 학습

HCE Loss = MSE  +  λ × Σ BCE(GO_node_k)
                     ↑
           계층 정보로 regularization → 붕괴 방지 + 해석 가능성 향상
```

---

## 핵심 결과

### Norman 데이터셋 (K562 세포주, 91,205 cells, 5,045 genes, 254 조건)

| 모델 | Best Val Pearson | ep15 Val Pearson | 비고 |
|------|:---:|:---:|------|
| GEARS baseline | 0.692 | **0.005** ❌ | 학습 붕괴 |
| **GEARS + HCE (λ=0.3)** | **0.817** | **0.700** ✅ | **붕괴 방지** |
| scGPT_brain + GEARS + HCE | 0.809 | 0.807 ✅ | scGPT 초기화 실험 |

> **HCE의 핵심 효과**: GEARS baseline은 ep15에서 Pearson이 0.005로 붕괴,
> HCE 추가 시 ep15에서도 0.700 유지 (붕괴 없음). Best Pearson도 0.692 → 0.817 향상.

### scGPT 통합 실험 (도메인 전이 분석)

| 모델 | Test Pearson | 비고 |
|------|:---:|------|
| scGPT_brain + HCE (frozen) | 0.193 | 뇌 사전학습 → K562 도메인 불일치 |
| scGPT_brain + HCE (last 2 layers unfrozen) | 0.251 | 부분 해동으로 +30% 향상 |
| **scGPT + GEARS + HCE** | **0.685** | scGPT 임베딩으로 GEARS 초기화 |

### OOD (Out-of-Distribution) 벤치마크 (K562 GWPS)

| 모델 | Pearson | GO AUROC | 비고 |
|------|:---:|:---:|------|
| λ=0.0 (pure MSE) | 0.451 | 0.556 | 기준선 |
| **λ=0.1 (HCE)** | **0.452** | **0.687** | **GO 경로 예측 +24%** |

> HCE는 회귀 성능(Pearson)을 유지하면서 생물학적 경로 예측(AUROC)을 대폭 향상.

---

## 구현 구조

```
HCE/
├── ontology.py              # GO/Cell Ontology DAG 구조 표현
├── loss.py                  # HierarchicalCrossEntropyLoss (HCE 손실 함수)
├── model.py                 # HCEPerturbationPredictor (단독 사용 모델)
├── config.py                # 경로 설정 (환경변수 오버라이드 지원)
│
├── gears_hce.py             # GEARS 서브클래싱 통합 (핵심 파일)
├── gears_norman_hce.py      # Norman 데이터셋 GEARS+HCE 학습 스크립트
├── gears_norman_baseline.py # 비교용 GEARS baseline
│
├── scgpt_norman_hce.py      # scGPT_brain frozen + HCE (도메인 전이 실험)
├── scgpt_norman_finetune.py # scGPT_brain 부분 해동 + HCE (Task 2)
├── scgpt_gears_hce.py       # scGPT 임베딩 → GEARS 초기화 + HCE (Task 3)
│
├── benchmark_ood.py         # OOD 벤치마크 (lambda 스윕)
├── benchmark_full_go.py     # Full GO (3,693 terms) 벤치마크
├── msigdb_ontology.py       # MSigDB Hallmark (44 terms) 온톨로지
├── go_ontology_full.py      # 실제 go-basic.obo 기반 전체 GO
├── data_replogle.py         # Replogle K562 데이터 로더
│
├── jacobian/                # scGPT_brain + HCE Jacobian 분석
│   ├── step1_finetune_hce.py    # scGPT_brain + brain cell ontology 파인튜닝
│   ├── step2_hce_jacobian.py    # ∂P(GO_node)/∂(gene) Jacobian 계산
│   ├── step3_visualize.py       # 계층별 attribution 시각화
│   └── README.md
│
├── results/                 # 실험 결과 JSON
└── demo.py                  # 전체 파이프라인 데모
```

---

## 핵심 구성요소

### 1. OntologyDAG (`ontology.py`)

GO 또는 Cell Ontology를 방향 비순환 그래프(DAG)로 표현. 자식 노드가 양성이면 모든 조상 노드도 양성으로 전파 (GO의 "is_a" 관계 반영).

```python
from HCE.ontology import OntologyDAG

dag = OntologyDAG()
dag.add_edge("apoptosis", "cell_death")          # apoptosis is_a cell_death
dag.add_edge("cell_death", "cellular_process")

# 양성 경로의 모든 조상 자동 전파
ancestors = dag.get_ancestors("apoptosis")
# → {"apoptosis", "cell_death", "cellular_process", "biological_process"}
```

구현된 온톨로지:
- `build_k562_go_ontology()` — K562 관련 GO 경로 **16 리프, 29 노드**
- `build_hallmark_ontology()` — MSigDB Hallmark **44 리프, 52 노드**
- `build_go_dag_from_obo()` — 실제 GO (go-basic.obo): **3,693 유효 텀, 1,121 리프**

### 2. HCE 손실 함수 (`loss.py`)

```
L_total = L_MSE  +  λ × L_HCE

L_MSE:   발현량 변화 예측 (회귀)
L_HCE:   GO 경로 multi-label 계층 분류
           → P(apoptosis) ≤ P(cell_death) ≤ P(cellular_process)
           → 틀려도 "큰 그림은 맞춤" → OOD 강건성 향상
```

| λ 값 | 효과 |
|------|------|
| 0.0 | 순수 MSE (기준선) |
| **0.1** | **최적: Pearson 유지 + GO AUROC +24%** |
| 0.3 | Norman 데이터 최적: 학습 붕괴 방지 |
| >0.5 | 회귀 성능 저하 시작 |

### 3. GEARS 서브클래싱 (`gears_hce.py`)

기존 GEARS 코드를 **수정하지 않고** 서브클래싱으로 HCE 통합. 기존 학습 인터페이스 그대로 사용 가능.

```python
# 기존 GEARS와 동일한 인터페이스
gears = GEARSWithHCE(pert_data, device='cuda')
gears.model_initialize_hce(
    hidden_size=64,
    lambda_hce=0.3,    # ← HCE 추가 파라미터
)
gears.train(epochs=15)
```

내부 변경 요약:

| 항목 | 기존 GEARS | GEARSWithHCE |
|------|------|------|
| 모델 | `GEARS_Model` | `GEARSModelWithHCE` (GO head 추가) |
| 손실 | `MSE + direction` | `MSE + direction + λ·HCE` |
| GO 라벨 | 없음 | 섭동 유전자 → GO 경로 실시간 생성 |
| 출력 | `pred_expr` | `(pred_expr, go_logits)` |

---

## 실행 방법

### 환경 설정

```bash
# 주 환경 (GEARS + 기본 실험)
conda activate gears2

# scGPT 통합 실험
conda activate scgpt
```

### GEARS + HCE 학습 (Norman 데이터셋)

```bash
cd /data2/Atlas_Normal
python -m HCE.gears_norman_hce      # GEARS + HCE (λ=0.3, epochs=15)
python -m HCE.gears_norman_baseline # 비교용 baseline
```

### scGPT 통합 실험

```bash
# Task 2: scGPT_brain 부분 해동 + HCE
conda run -n scgpt python -m HCE.scgpt_norman_finetune

# Task 3: scGPT 임베딩으로 GEARS 초기화 + HCE
conda run -n scgpt python -m HCE.scgpt_gears_hce
```

### OOD 벤치마크

```bash
python -m HCE.benchmark_ood        # Hallmark 온톨로지, lambda 스윕
python -m HCE.benchmark_full_go    # Full GO (3,693 terms)
```

### Jacobian 분석 파이프라인

```bash
python -m HCE.jacobian.step1_finetune_hce   # scGPT_brain + HCE 파인튜닝
python -m HCE.jacobian.step2_hce_jacobian   # ∂P(GO_node)/∂(gene) 계산
python -m HCE.jacobian.step3_visualize      # 시각화
```

### Python API

```python
import sys
sys.path.insert(0, "/data2/Atlas_Normal")
from HCE import HCEPerturbationPredictor
from HCE.data_replogle import ReplogleDataset, build_k562_go_ontology

dataset = ReplogleDataset("K562_gwps_raw_bulk.h5ad", gene_subset=2000)
train_ds, val_ds, test_ds = dataset.get_splits()

dag, term_to_idx = build_k562_go_ontology()
model = HCEPerturbationPredictor(
    n_genes=2000, n_go_terms=len(term_to_idx),
    ontology=dag, go_term_to_idx=term_to_idx,
    lambda_cls=0.1,   # 최적값 (OOD 벤치마크 기준)
)
loss, info = model.compute_loss(expr, pert_mask, delta_expr, go_labels)
```

---

## Jacobian 분석 — 뇌 세포 유형별 GO 경로 attribution

**목표**: scGPT_brain + HCE를 뇌 단일세포 데이터에 파인튜닝 → `∂P(GO_level_k)/∂(gene_expr)` Jacobian으로 어떤 유전자가 어떤 GO 경로 레벨에서 중요한지 정량화.

| 항목 | 값 |
|------|---|
| 사전학습 | scGPT_brain (13.2M 뇌 세포) |
| 파인튜닝 데이터 | Fetal brain atlas (116,529 cells) |
| 세포 유형 | 4종: RG / Neuroblast / Ext / Inh |
| Step 1 분류 정확도 | **67.5%** (Ext=95.6%, Neuroblast=66.7%) |
| Jacobian 단조성 | Leaf → 부모: **1.000 ✅** (HCE 효과 완벽 검증) |

자세한 결과: [`jacobian/README.md`](jacobian/README.md)

---

## 환경

| 항목 | 버전 |
|------|------|
| Python | 3.10 (scgpt) / 3.11 (gears2) |
| PyTorch | 2.10.0+cu128 |
| GEARS | snap-stanford/GEARS |
| scGPT | 0.2.1 |
| scanpy | 1.11.5 |
| GPU | CUDA 12.8 |

---

## 향후 과제

1. **Multi-label sigmoid 전환**: softmax → per-node sigmoid → 루트 단조성 해결
2. ✅ **scGPT_brain + 부분 해동 (Task 2)**: Pearson 0.193 → 0.251 (+30%)
3. ✅ **scGPT + GEARS + HCE (Task 3)**: Test Pearson 0.685 (val 0.809, 매우 안정)
4. **Integrated Gradients**: Jacobian 대신 더 안정적인 attribution 방법
5. **Norman + Adamson 통합 평가**: 공식 GEARS 평가 지표 (mean Pearson by split type)
6. **scGPT_human 도메인 전이**: brain → human으로 올바른 도메인 실험
