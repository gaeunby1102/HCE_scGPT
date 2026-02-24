"""
config.py
---------
경로 설정. 환경변수로 오버라이드 가능.

사용 예시:
    export HCE_DATA_ROOT=/your/data/path
    export HCE_K562_DATA=/your/path/K562_gwps_raw_bulk.h5ad
    export HCE_SCGPT_BRAIN=/your/path/scGPT_brain
"""
import os

# ── 루트 디렉토리 ──────────────────────────────────────────────────────
# 데이터/모델 저장소 기본 경로 (환경변수로 오버라이드)
DATA_ROOT    = os.environ.get("HCE_DATA_ROOT",    "./data")
RESULTS_ROOT = os.environ.get("HCE_RESULTS_ROOT",
                               os.path.join(os.path.dirname(__file__), "results"))

# ── 데이터 파일 ────────────────────────────────────────────────────────
# Replogle K562 GWPS perturbation data
K562_DATA = os.environ.get(
    "HCE_K562_DATA",
    os.path.join(DATA_ROOT, "K562_gwps_raw_bulk.h5ad"),
)

# Fetal brain single-cell atlas (scGPT Jacobian 분석용)
BRAIN_ATLAS = os.environ.get(
    "HCE_BRAIN_ATLAS",
    os.path.join(DATA_ROOT, "brain_atlas.h5ad"),
)

# ── GEARS 데이터 ───────────────────────────────────────────────────────
GEARS_DATA_DIR = os.environ.get(
    "HCE_GEARS_DATA",
    os.path.join(DATA_ROOT, "gears"),
)
GEARS_SAVE_NORMAN = os.environ.get(
    "HCE_GEARS_SAVE_NORMAN",
    os.path.join(DATA_ROOT, "gears_hce_norman"),
)
GEARS_SAVE_BASELINE = os.environ.get(
    "HCE_GEARS_SAVE_BASELINE",
    os.path.join(DATA_ROOT, "gears_baseline_norman"),
)

# ── GO 온톨로지 ────────────────────────────────────────────────────────
GO_OBO   = os.environ.get("HCE_GO_OBO",   os.path.join(DATA_ROOT, "go-basic.obo"))
GENE2GO  = os.environ.get("HCE_GENE2GO",  os.path.join(DATA_ROOT, "gene2go_all.pkl"))
GO_CACHE = os.environ.get("HCE_GO_CACHE", os.path.join(DATA_ROOT, "go_dag_min50_max2000_BP_CC_MF.pkl"))

HALLMARK_CACHE = os.environ.get(
    "HCE_HALLMARK_CACHE",
    os.path.join(DATA_ROOT, "hallmark_genes.json"),
)

# ── scGPT 모델 ─────────────────────────────────────────────────────────
# scGPT_brain checkpoint (bowang-lab/scGPT, brain pretrained)
SCGPT_BRAIN_DIR = os.environ.get(
    "HCE_SCGPT_BRAIN",
    os.path.join(DATA_ROOT, "scGPT_brain"),
)

# ── 결과 저장 ──────────────────────────────────────────────────────────
JACOBIAN_RESULTS = os.environ.get(
    "HCE_JACOBIAN_RESULTS",
    os.path.join(os.path.dirname(__file__), "jacobian", "results"),
)
