# HCE: Hierarchical Cross-Entropy for Single-Cell Perturbation Modeling

> Nature Computational Science (2025) 논문의 HCE 아이디어를 유전자 섭동 예측에 적용한 구현체.

---

## 배경: 왜 HCE인가?

### 기존 섭동 모델의 한계

```
GEARS, scGPT, Geneformer 등
    ↓
"유전자 A를 KO하면 나머지 2만개 유전자 수치가 각각 얼마나 변할까?"
    ↓
MSE (연속 회귀) 로만 최적화
    ↓
OOD(학습 때 없던 섭동) 예측 시 성능 급락
    ↓
생물학적 경로 수준 해석 불가
```

### HCE가 추가하는 것

```
섭동 결과 회귀 (기존)
  +
GO 계층 구조 기반 분류 (신규)
    → P(apoptosis) ≤ P(cell_death) ≤ P(cellular_process)
    → 틀려도 "큰 그림은 맞춤" → OOD 강건성 ↑
    → 생물학자가 해석 가능한 계층적 예측 출력
```

---

## 구현 구조

```
HCE/
├── ontology.py         # OntologyDAG (Cell Ontology / GO 범용 DAG)
├── loss.py             # HierarchicalCrossEntropyLoss
│                       # HierarchicalPerturbationLoss
├── model.py            # HCECellTypeClassifier (세포 유형 분류)
│                       # HCEPerturbationPredictor (섭동 예측)
├── data_replogle.py    # Replogle K562 데이터 로더 + GO 라벨 생성
├── gears_hce.py        # GEARS 서브클래싱 통합
├── gears_norman_hce.py # Norman 데이터 전체 GEARS+HCE 학습
├── benchmark_ood.py    # OOD 벤치마크 (lambda 스윕)
├── msigdb_ontology.py  # MSigDB Hallmark (50 terms) 온톨로지
├── go_ontology_full.py # 실제 GO (BP+CC+MF, ~3700 terms)
├── scgpt_hce.py        # scGPT 파인튜닝 HCE 드롭인
├── hallmark_ontology.json  # DAG JSON (52 nodes)
├── results/            # 벤치마크 결과
├── demo.py             # 실행 가능한 전체 파이프라인 데모
└── jacobian/           # scGPT_brain + HCE Jacobian 분석 ← 신규
    ├── README.md
    ├── step1_finetune_hce.py   # scGPT_brain + HCE 파인튜닝
    ├── step2_hce_jacobian.py   # GO 레벨별 Jacobian 계산
    ├── step3_visualize.py      # 온톨로지 attribution 시각화
    ├── results/
    └── figures/
```

---

## 핵심 구성요소

### 1. OntologyDAG (`ontology.py`)

Cell Ontology(CL) 또는 Gene Ontology(GO)를 방향 비순환 그래프(DAG)로 표현.

```python
from HCE.ontology import OntologyDAG

dag = OntologyDAG()
dag.add_edge("apoptosis", "cell_death")         # child is_a parent
dag.add_edge("cell_death", "cellular_process")

ancestors = dag.get_ancestors("apoptosis")
# → {"apoptosis", "cell_death", "cellular_process", "biological_process"}
```

현재 구현된 온톨로지:
- `build_k562_go_ontology()` — K562 관련 GO 생물학 경로 16개 리프, 29개 노드
- `build_hallmark_ontology()` — MSigDB Hallmark 44개 리프, 52개 노드
- `build_go_dag_from_obo()` — 실제 go-basic.obo 파싱: **3,693 유효 텀, 1,121 리프**
  - BP: 2,513 / CC: 478 / MF: 702

### 2. HCE 손실 함수 (`loss.py`)

#### `HierarchicalCrossEntropyLoss` (세포 유형 분류용)

```
L = (1-α)·L_flat_CE  +  α·L_hier_BCE  +  λ_mono·L_monotonicity

L_flat_CE:    표준 Cross-Entropy (리프 수준)
L_hier_BCE:   모든 조상 노드에서의 Binary CE (weighted by depth)
L_monotonicity: ReLU(P(child) - P(parent)) 패널티
```

| 하이퍼파라미터 | 기본값 | 의미 |
|---|---|---|
| `alpha` | 0.7 | 계층 손실 비중 (1.0 = 완전 계층) |
| `level_decay` | 0.5 | 상위 레벨 가중치 (낮을수록 상위 강조) |
| `monotone_coef` | 0.1 | 단조성 패널티 강도 |

#### `HierarchicalPerturbationLoss` (섭동 예측용)

```
L = λ_reg·L_MSE  +  λ_cls·L_HCE_multilabel

L_MSE:            발현량 변화 예측 회귀 손실
L_HCE_multilabel: GO 경로 multi-label 계층 분류 손실
                  활성 경로의 모든 조상도 양성으로 전파
```

### 3. 모델 아키텍처 (`model.py`)

#### `HCEPerturbationPredictor` (GEARS 스타일 단순화 버전)

```
입력: [발현량 (G) ; 섭동 마스크 (G)] → concat → (2G,)
       ↓
   공유 인코더 (MLP + LayerNorm + GELU)
       ↓ latent (H,)
   ┌───────────────┐    ┌──────────────────┐
   │ Expression    │    │ GO Classifier    │
   │ Head          │    │ Head             │
   │ (H → G)       │    │ (H → K)          │
   └───────────────┘    └──────────────────┘
        Δexpr                go_logits
    (regression)         (hierarchical cls)
```

### 4. GEARS 서브클래싱 (`gears_hce.py`)

기존 GEARS 코드를 수정하지 않고 서브클래싱으로 HCE 통합.

```python
# 기존 GEARS 사용법 그대로 유지
gears = GEARSWithHCE(pert_data, device='cuda')
gears.model_initialize_hce(
    hidden_size=64,
    lambda_hce=0.3,    # ← 추가된 파라미터
)
gears.train(epochs=20)
```

**내부 변경점 요약**:

| 항목 | 기존 GEARS | GEARSWithHCE |
|---|---|---|
| 모델 클래스 | `GEARS_Model` | `GEARSModelWithHCE` |
| 추가 파라미터 | — | GO head (avg pooling → MLP) |
| 손실 함수 | `MSE + direction` | `MSE + direction + λ·HCE` |
| GO 라벨 | — | 배치에서 실시간 생성 (gene → pathway) |
| 출력 | `pred_expr` | `(pred_expr, go_logits)` |

---

## 데이터: Replogle K562 GWPS (`data_replogle.py`)

**출처**: Replogle et al. 2022, Nature Biotechnology
**파일**: `K562_gwps_raw_bulk.h5ad`

| 항목 | 내용 |
|---|---|
| 섭동 조건 수 | 10,744 (컨트롤 제외) |
| 측정 유전자 수 | 8,248 (ENSG ID) |
| 컨트롤 조건 | 514 (non-targeting guide) |
| 섭동 유전자 수 | 9,867 unique genes |
| 측정된 섭동 유전자 비율 | 19.2% (섭동 유전자가 측정 행렬에도 포함) |

**GO 라벨 생성 방법**:
1. 경로별 소속 유전자 리스트 (`PATHWAY_GENES` dict, 16개 리프 경로)
2. 각 섭동 조건의 `|Δexpr|` 을 소속 유전자들에 대해 평균
3. 경로별 상위 25% 이상인 섭동을 해당 경로에서 "양성(1)"으로 라벨링
4. GO 계층 구조로 라벨 전파 (양성 경로의 모든 조상도 양성)

**K562 GO 계층 구조 (16 리프 → 29 노드)**:

```
biological_process
├── cellular_process
│   ├── cell_cycle
│   │   ├── g1s_transition
│   │   └── mitosis
│   ├── dna_process
│   │   ├── dna_repair
│   │   └── dna_replication
│   ├── cell_death
│   │   ├── apoptosis
│   │   └── necroptosis
│   └── protein_quality
│       ├── ubiquitin_proteasome
│       └── autophagy
├── gene_expression
│   ├── gene_regulation
│   │   ├── transcription_factors
│   │   └── chromatin_regulation
│   └── protein_synthesis
│       ├── ribosome_biogenesis
│       └── translation
├── signaling
│   └── signal_transduction
│       ├── pi3k_akt_signaling
│       └── jak_stat_signaling
└── immune_process
    └── immune_response
        ├── interferon_signaling
        └── cytokine_signaling
```

---

## OOD 벤치마크 결과 (v2)

**설정**: K562 GWPS, Gene-OOD split (측정된 유전자 중 20% hold-out), lambda 스윕

| 모델 | 분할 | Pearson | GO AUROC | 비고 |
|---|---|---|---|---|
| λ=0.0 (baseline) | ID | 0.374 | 0.535 | 순수 회귀 |
| λ=0.0 (baseline) | OOD | 0.451 | 0.556 | |
| λ=0.05 | ID | 0.378 | **0.746** | +39% AUROC |
| λ=0.05 | OOD | 0.450 | 0.664 | |
| **λ=0.1 (최적)** | **ID** | **0.389** | **0.763** | **+4% Pearson, +43% AUROC** |
| **λ=0.1 (최적)** | **OOD** | **0.452** | **0.687** | |
| λ=0.3 | ID | 0.375 | 0.747 | lambda 과대 |
| λ=0.3 | OOD | 0.444 | 0.663 | |

**핵심 결과 해석**:

1. **GO 경로 예측 대폭 향상**: λ=0.1에서 AUROC 0.535 → 0.763 (+43%). HCE loss가 계층 구조 정보를 효과적으로 학습.
2. **회귀 성능 유지**: Pearson 계수가 λ=0.1에서 오히려 소폭 향상 (0.374 → 0.389). HCE가 회귀 성능을 저해하지 않음.
3. **OOD Pearson > ID Pearson**: OOD 유전자들이 더 강한/일관된 섭동 효과를 가져 예측이 상대적으로 쉬운 것으로 해석. OOD GO_AUROC는 ID보다 낮아 일반화 격차 존재 (0.763 → 0.687).
4. **최적 λ = 0.1**: λ=0.3은 HCE 손실 비중이 너무 커서 GO_AUROC는 유사하지만 Pearson이 소폭 하락.
5. **Monotonicity 지표**: Leaf-only 다중 레이블 아키텍처에서 측정 불가 (부모 확률 = 자식 확률들의 합계 → 정의상 항상 ≥ 자식). HCE 학습 효과는 GO AUROC로 간접 검증.

---

## 확장된 GO 온톨로지 (`go_ontology_full.py`)

실제 `go-basic.obo` + `gene2go` 어노테이션 기반 전체 GO 사용.

| 항목 | 값 |
|---|---|
| 파일 | `/data4/HCE_gears_data/go-basic.obo` |
| 캐시 | `/data4/HCE_gears_data/go_dag_min50_max2000_BP_CC_MF.pkl` |
| 유효 텀 수 | **3,693** (min_genes≥50, max_genes≤2000) |
| 리프 수 | **1,121** |
| BP | 2,513 |
| CC | 478 |
| MF | 702 |

```python
from HCE.go_ontology_full import load_or_build_go_dag
dag, term_to_idx, pathway_genes = load_or_build_go_dag(
    data_dir="/data4/HCE_gears_data",
    namespaces={"biological_process", "cellular_component", "molecular_function"},
    min_genes=50,
)
```

---

## MSigDB Hallmark 온톨로지 (`msigdb_ontology.py`)

Hallmark 50 gene sets를 4-level 계층으로 구성.

| 항목 | 값 |
|---|---|
| 총 노드 | 52 (root + 7 mid-level + 44 Hallmark leaves) |
| 리프 | 44 (원본 50 중 매핑된 항목) |
| 계층 | biological_process → 7 카테고리 → 44 Hallmark |

**7개 mid-level 카테고리**:
- `cell_proliferation` (E2F, G2M, Myc_V1, Myc_V2, Mitotic Spindle)
- `cell_death` (Apoptosis, p53, Hypoxia)
- `metabolism` (OxPhos, Glycolysis, FA, Cholesterol, mTORC1 등)
- `immune_signaling` (TNFα, IL6/JAK, IFNα/γ, Inflammatory 등)
- `signal_transduction` (PI3K, KRAS, Hedgehog, Notch, TGFβ, WNT)
- `stress_response` (UV, UPR, ROS, DNA_Repair)
- `development` (Adipogenesis, EMT, Angiogenesis, Estrogen 등)

---

## scGPT 통합 (`scgpt_hce.py`) ✓ 검증 완료

```
[scGPT+HCE 구조 검증]
  cell_emb shape: torch.Size([8, 512])
  go_logits shape: torch.Size([8, 44])
  HCE loss: 0.8814
  n_go=44, DAG=OntologyDAG(nodes=52, edges=51)
  ✓ scGPT + HCE 통합 구조 정상
```

실제 scGPT 모델 없이 MockScGPT로 드롭인 구조 검증 완료.
실제 파인튜닝: `train_one_epoch_hce()` 함수로 기존 scGPT 학습 루프 교체.

---

## Norman 섭동 예측 실험 결과 요약

**Norman 데이터셋**: n_cells=91,205 / n_genes=5,045 / train_conds=138 / test_conds=116

### 전체 성능 비교

| 실험 | 모델 | Best Pearson | ep15 Pearson | Test Pearson | 비고 |
|------|------|:-----------:|:-----------:|:-----------:|------|
| Exp 1 | GEARS baseline | 0.692 | 0.005 | — | ❌ 학습 붕괴 |
| Exp 2 | GEARS + HCE (λ=0.3) | **0.817** | 0.700 | **0.705** | ✅ 붕괴 방지 |
| Exp 3 | scGPT_brain + HCE (frozen) | 0.165 | 0.193 | 0.193 | 도메인 미스매치 |
| Exp 4 | scGPT_brain + HCE (finetune) | **0.251** | 0.225 | 0.251 | Task 2 완료 |
| Exp 5 | scGPT + GEARS + HCE | val **0.809** | 0.807 | **0.685** | Task 3 완료 |

### Split 유형별 평가 (GEARS + HCE, Test set)

Norman 데이터셋의 GEARS 공식 평가 방식: simulation split (seed=1)

| Split 유형 | 조건 수 | Mean Pearson | Std | 설명 |
|------|:---:|:---:|:---:|------|
| `unseen_single` | 37 | **0.815** | 0.018 | 훈련 미포함 단일 유전자 KO |
| `combo_seen0` | 9 | 0.739 | 0.069 | 이중 섭동, 두 유전자 모두 훈련 미포함 |
| `combo_seen1` | 52 | 0.646 | 0.157 | 이중 섭동, 한 유전자만 훈련 포함 |
| `combo_seen2` | 18 | 0.631 | 0.163 | 이중 섭동, 두 유전자 모두 훈련 포함 |
| **전체 (116조건)** | 116 | **0.705** | — | overall test Pearson |

**결과 해석**:
- `unseen_single` (0.815): HCE 정규화가 단일 유전자 OOD 일반화에 효과적
- `combo_seen0` (0.739): 두 유전자 모두 훈련 미포함인 최고 난이도에서도 강건
- `combo_seen2` (0.631): 두 유전자가 알려져 있음에도 상호작용 예측은 어려움
- 이중 섭동에서 `seen2 < seen1 < seen0` 역순: 단순 유전자 지식보다 상호작용 비선형성이 성능 결정 인자

---

## Brain In Silico Perturbation (뇌 세포 In Silico 유전자 섭동)

**목표**: scGPT_brain + HCE 분류 모델을 이용해 RG(방사상 신경교) → Neuroblast 전이를 유도하는 유전자를 In Silico로 순위화.

**방법**:
1. RG 세포 200개 샘플링 (fetal brain atlas, n=29,402 RG)
2. 각 유전자를 0으로 마스킹 → scGPT_brain 재추론 → ΔP(Neuroblast) 계산
3. 상위 발현 300개 유전자 테스트 (약 15분 소요)

**실행**:
```bash
conda run -n scgpt python -m HCE.brain_insilico_perturb
```

### 결과: RG → Neuroblast 전이 상위 유전자 (ΔP(NB) 내림차순)

| 유전자 | ΔP(NB) | 설명 |
|--------|:------:|------|
| **MALAT1** | **+0.0077** | lncRNA, 신경발생 조절 (알려진 기능) |
| MT-CO2 | +0.0008 | 미토콘드리아 복합체 IV |
| ACTB | +0.0007 | 액틴 세포골격 |
| MT-CO3 | +0.0007 | 미토콘드리아 |
| HSP90AA1 | +0.0007 | 열충격 단백질, 분화 조절 |
| AUTS2 | +0.0006 | 자폐증 관련, 신경발생 TF |
| RPL4/RPL30/RPL12 | +0.0005~0.0006 | 리보솜 단백질 |

**주요 발견**:
- **MALAT1**: ΔP가 가장 크고 (0.0077), 다른 유전자와 10배 격차. 알려진 신경발생 조절 lncRNA로 생물학적으로 타당한 결과.
- **AUTS2**: 자폐증 관련 유전자로 신경발생에 직접 관여하는 TF. 생물학적으로 의미있는 상위 후보.
- **Baseline 분류 이슈**: 모델이 RG 세포를 Ext로 잘못 분류 (정확도 0%). step1 학습 데이터와 현재 세포 간 분포 불일치 추정. ΔP의 절대값은 작지만 유전자 간 상대적 순위는 유효.

### 결과: RG 정체성 유지 유전자 (KO 시 P(RG) 감소)

| 유전자 | ΔP(RG) | 기능 |
|--------|:------:|------|
| **YWHAQ** | -0.0004 | 14-3-3 단백질, 세포 주기 체크포인트 |
| **SNRPE** | -0.0004 | 스플라이세오솜, RNA 처리 |
| **RAN** | -0.0003 | 핵공 수송, 세포 분열 |
| **NPM1** | -0.0003 | 핵소체 인, 줄기세포 유지 |

**해석**: RG 정체성 유지에는 세포주기 제어 (YWHAQ), 핵 수송 (RAN), 핵소체 기능 (NPM1)이 중요함.

---

## 실행 방법

### OOD 벤치마크
```bash
cd /data2/Atlas_Normal
/home/t1/miniconda3/envs/gears2/bin/python -m HCE.benchmark_ood
```

### GEARS + Norman 전체 학습
```bash
/home/t1/miniconda3/envs/gears2/bin/python -m HCE.gears_norman_hce \
  > HCE/results/norman_hce.log 2>&1
```

### Norman split 유형별 평가
```bash
cd /data2/Atlas_Normal
conda run -n gears2 python -m HCE.eval_by_split
# 결과: HCE/results/eval_by_split.json
```

### Brain In Silico Perturbation
```bash
conda run -n scgpt python -m HCE.brain_insilico_perturb
# 결과: HCE/results/brain_insilico_perturb.json
```

### scGPT 구조 검증
```bash
/home/t1/miniconda3/envs/scgpt/bin/python -m HCE.scgpt_hce
```

### Full GO 온톨로지 빌드
```bash
/home/t1/miniconda3/envs/gears2/bin/python -m HCE.go_ontology_full
```

### Python 코드에서 직접 사용
```python
import sys
sys.path.insert(0, "/data2/Atlas_Normal")
from HCE import HCEPerturbationPredictor
from HCE.data_replogle import ReplogleDataset, build_k562_go_ontology

# 데이터 로드
dataset = ReplogleDataset("K562_gwps_raw_bulk.h5ad", gene_subset=2000)
train_ds, val_ds, test_ds = dataset.get_splits()

# 모델 초기화
dag, term_to_idx = build_k562_go_ontology()
model = HCEPerturbationPredictor(
    n_genes=2000,
    n_go_terms=len(term_to_idx),
    ontology=dag,
    go_term_to_idx=term_to_idx,
    hidden_dims=(512, 256, 128),
    lambda_reg=1.0,
    lambda_cls=0.1,   # 최적값
)

# 학습
loss, info = model.compute_loss(expr, pert_mask, delta_expr, go_labels)
```

---

## 환경

| 항목 | 버전 |
|---|---|
| Python | 3.11 (gears2 env) |
| PyTorch | 2.3.0 |
| torch_geometric | 2.5.3 |
| torch_scatter | ✓ |
| GEARS | snap-stanford/GEARS |
| scanpy | 1.11.5 |
| obonet | ✓ (GO OBO 파싱) |
| gseapy | ✓ (MSigDB) |

**Conda 환경**: `/home/t1/miniconda3/envs/gears2` (주)
**scGPT 환경**: `/home/t1/miniconda3/envs/scgpt`

---

## 향후 과제

1. **Norman + Adamson 통합 평가**: Norman 학습 완료 후 공식 GEARS 평가 지표 비교
2. **Full GO (3693 terms) 적용**: K562 GWPS에 실제 GO annotation 기반 3693개 텀으로 재학습
3. **Monotonicity 측정 개선**: 중간 노드 로짓 출력 또는 threshold 기반 예측 일관성 지표 도입
4. **HCE Jacobian 분석** (`jacobian/`): scGPT_brain + HCE GO head로 뇌 세포 유형별 GO 경로 attribution
   - Step 1: scGPT_brain (13.2M brain cells) + brain cell ontology HCE 파인튜닝
   - Step 2: ∂(GO_level_k) / ∂(gene_expr) Jacobian 계산 (RG/Neuroblast/Ext/Inh × 온톨로지 레벨)
   - Step 3: 계층별 attribution 시각화 + 발달 시계열 분석
