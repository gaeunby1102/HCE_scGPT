"""
scgpt_norman_hce.py
-------------------
scGPT_brain (frozen encoder) + HCE loss → Norman 섭동 예측.

아키텍처:
  scGPT_brain(ctrl_expr) → cell_emb (512D)         [frozen]
  scGPT_brain.encoder(pert_gene) → pert_emb (512D)  [frozen]
  cat([cell_emb, pert_emb]) → predictor → Δexpr
                            → go_head   → GO logits → HCE loss

비교 대상:
  GEARS baseline:  best Pearson=0.692, ep15=0.005 (붕괴)
  GEARS+HCE:       best Pearson=0.817, ep15=0.70  (안정)

실행:
    python -m HCE.scgpt_norman_hce
"""
from __future__ import annotations
import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import scipy.sparse

import HCE.config as cfg

SCGPT_DIR   = cfg.SCGPT_BRAIN_DIR
GEARS_DIR   = cfg.GEARS_DATA_DIR
RESULT_DIR  = cfg.RESULTS_ROOT
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEQ     = 1200
N_BINS      = 51
EPOCHS      = 15
BATCH       = 32
LR          = 3e-4
LAMBDA_HCE  = 0.1

LOG_PATH    = os.path.join(RESULT_DIR, "scgpt_norman_hce.log")
RESULT_PATH = os.path.join(RESULT_DIR, "scgpt_norman_hce.json")


# ── scGPT_brain 로드 ──────────────────────────────────────────────────
def load_scgpt():
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab

    vocab = GeneVocab.from_file(os.path.join(SCGPT_DIR, "vocab.json"))
    args  = json.load(open(os.path.join(SCGPT_DIR, "args.json")))

    model = TransformerModel(
        ntoken          = len(vocab),
        d_model         = args["embsize"],
        nhead           = args["nheads"],
        d_hid           = args["d_hid"],
        nlayers         = args["nlayers"],
        nlayers_cls     = args.get("n_layers_cls", 3),
        n_cls           = 1,
        vocab           = vocab,
        dropout         = args["dropout"],
        pad_token       = args["pad_token"],
        pad_value       = args["pad_value"],
        do_mvc          = False,
        do_dab          = False,
        use_batch_labels= False,
        input_emb_style = args["input_emb_style"],
        n_input_bins    = args["n_bins"],
        cell_emb_style  = "cls",
        use_fast_transformer = args.get("fast_transformer", True),
        pre_norm        = False,
    )
    ckpt = torch.load(os.path.join(SCGPT_DIR, "best_model.pt"), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    return model, vocab, args


# ── Norman 데이터셋 ───────────────────────────────────────────────────
class NormanScGPTDataset(Dataset):
    """
    Norman PertData → scGPT 입력 형식.

    각 샘플: 섭동 세포의 발현값을 scGPT 형식으로 변환.
    입력:   (gene_ids, values)  ← 세포 발현량
    pert:   pert_gene_id        ← 섭동 유전자 vocab ID
    target: delta_expr          ← 섭동 효과 (perturbed - ctrl_mean)
    label:  go_label            ← Hallmark GO 라벨
    """
    def __init__(self, adata, vocab, pathway_genes, term_to_idx,
                 max_seq=MAX_SEQ, n_bins=N_BINS):
        pad_id = vocab["<pad>"]
        # Norman var_names = Ensembl ID → gene_name 컬럼 사용
        if "gene_name" in adata.var.columns:
            gene_names = adata.var["gene_name"].tolist()
        else:
            gene_names = adata.var_names.tolist()

        # scGPT vocab에 있는 유전자만
        common_idx   = [i for i, g in enumerate(gene_names) if g in vocab]
        common_toks  = [vocab[gene_names[i]] for i in common_idx]
        print(f"  Norman 유전자-vocab 겹침: {len(common_idx)}/{len(gene_names)} "
              f"({len(common_idx)/len(gene_names):.1%})")

        # 발현 행렬
        X = adata.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        # ctrl mean (ctrl 조건 기준)
        ctrl_mask = (adata.obs["condition"] == "ctrl").values
        ctrl_mean = X[ctrl_mask].mean(axis=0)   # (n_genes,)
        print(f"  ctrl cells: {ctrl_mask.sum()}, pert cells: {(~ctrl_mask).sum()}")

        # 섭동 유전자 파싱
        def parse_pert_genes(cond: str):
            return [g for g in cond.split("+") if g != "ctrl"]

        # GO 라벨 사전 계산 (delta 기반)
        delta_all = X - ctrl_mean          # (N, n_genes)
        gene_to_idx_local = {g: i for i, g in enumerate(gene_names)}
        n_go = len(term_to_idx)

        go_scores = np.zeros((len(X), n_go), dtype=np.float32)
        for term, tidx in term_to_idx.items():
            pg = [g for g in pathway_genes.get(term, []) if g in gene_to_idx_local]
            if pg:
                gidx = [gene_to_idx_local[g] for g in pg]
                go_scores[:, tidx] = np.abs(delta_all[:, gidx]).mean(axis=1)

        thresholds = np.percentile(go_scores, 75, axis=0)
        go_labels_arr = (go_scores >= thresholds[None, :]).astype(np.float32)

        # scGPT 형식 변환 (lazy: __getitem__에서 처리)
        self.X           = X
        self.ctrl_mean   = ctrl_mean
        self.gene_names  = gene_names
        self.n_genes     = len(gene_names)
        self.common_idx  = np.array(common_idx, dtype=np.int32)
        self.common_toks = np.array(common_toks, dtype=np.int64)
        self.max_seq     = max_seq
        self.n_bins      = n_bins
        self.pad_id      = pad_id

        # 섭동 유전자 vocab ID (첫 번째 유전자만 사용, combo는 평균)
        self.pert_gene_ids = []
        for cond in adata.obs["condition"]:
            pert_gs = parse_pert_genes(cond)
            ids = [vocab[g] for g in pert_gs if g in vocab]
            self.pert_gene_ids.append(ids if ids else [pad_id])

        self.go_labels   = torch.tensor(go_labels_arr, dtype=torch.float32)
        self.conditions  = adata.obs["condition"].tolist()
        print(f"  GO 라벨 양성 비율: {go_labels_arr.mean():.3f}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 발현값 → scGPT 형식
        expr    = self.X[idx, self.common_idx]          # (n_common,)
        n_sel   = min(self.max_seq, len(expr))
        top_k   = np.argsort(expr)[-n_sel:][::-1]
        sel_ids = self.common_toks[top_k]
        sel_val = expr[top_k]

        sel_val = np.log1p(sel_val)
        max_v   = sel_val.max() + 1e-6 if len(sel_val) > 0 else 1e-6
        sel_val = sel_val / max_v
        binned  = np.floor(sel_val * (self.n_bins - 1)).astype(np.float32)

        pad_len   = self.max_seq - len(sel_ids)
        gene_ids  = np.concatenate([sel_ids, np.full(pad_len, self.pad_id, dtype=np.int64)])
        values    = np.concatenate([binned,  np.full(pad_len, -2.0, dtype=np.float32)])

        # 섭동 유전자: 여러 개면 vocab ID 첫 번째 사용 (combo는 mean pooling)
        pg_ids    = self.pert_gene_ids[idx]
        pert_id   = pg_ids[0]          # single or first gene of combo

        delta     = self.X[idx] - self.ctrl_mean

        return (
            torch.tensor(gene_ids, dtype=torch.long),
            torch.tensor(values,   dtype=torch.float32),
            torch.tensor(pert_id,  dtype=torch.long),
            torch.tensor(delta,    dtype=torch.float32),
            self.go_labels[idx],
        )


# ── 모델 ──────────────────────────────────────────────────────────────
class ScGPTNormanPredictor(nn.Module):
    """
    scGPT_brain(frozen) + perturbation predictor + GO head.

    forward(gene_ids, values, pad_mask, pert_gene_ids)
        cell_emb  = scGPT(gene_ids, values)        → (B, 512)
        pert_emb  = scGPT.encoder(pert_gene_ids)   → (B, 512)
        combined  = cat([cell_emb, pert_emb])       → (B, 1024)
        delta     = predictor(combined)             → (B, n_genes)
        go_logits = go_head(combined)               → (B, n_go)
    """
    def __init__(self, scgpt_model, n_genes, n_go, d_model=512):
        super().__init__()
        self.scgpt  = scgpt_model
        self.n_genes = n_genes

        self.predictor = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_genes),
        )
        self.go_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_go),
        )

    def forward(self, gene_ids, values, pad_mask, pert_gene_ids):
        with torch.no_grad():
            out      = self.scgpt(gene_ids, values,
                                  src_key_padding_mask=pad_mask,
                                  CLS=False, MVC=False, ECS=False)
            cell_emb = out["cell_emb"]                          # (B, 512)
            pert_emb = self.scgpt.encoder(pert_gene_ids)        # (B, 512)

        combined   = torch.cat([cell_emb, pert_emb], dim=-1)   # (B, 1024)
        delta_pred = self.predictor(combined)                   # (B, n_genes)
        go_logits  = self.go_head(combined)                     # (B, n_go)
        return delta_pred, go_logits


# ── 분할 ──────────────────────────────────────────────────────────────
def get_gene_ood_splits(dataset, set2conditions, seed=42):
    """
    GEARS의 set2conditions 활용 (train/val/test 조건 분리).
    ctrl 세포는 train에만 포함.
    """
    cond_to_split = {}
    for split_name, conds in set2conditions.items():
        for c in conds:
            cond_to_split[c] = split_name
    cond_to_split["ctrl"] = "train"

    train_idx, val_idx, test_idx = [], [], []
    for i, cond in enumerate(dataset.conditions):
        s = cond_to_split.get(cond, "train")
        if s == "train":
            train_idx.append(i)
        elif s in ("val", "dev"):
            val_idx.append(i)
        else:
            test_idx.append(i)

    # val이 없으면 train에서 10% 분할
    if not val_idx:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(len(train_idx) * 0.1))
        rng.shuffle(train_idx)
        val_idx   = train_idx[:n_val]
        train_idx = train_idx[n_val:]

    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))


# ── 평가 ──────────────────────────────────────────────────────────────
def evaluate(model, loader, pad_id):
    model.eval()
    pearsons = []
    with torch.no_grad():
        for gene_ids, values, pert_ids, delta, _ in loader:
            gene_ids = gene_ids.to(DEVICE)
            values   = values.to(DEVICE)
            pert_ids = pert_ids.to(DEVICE)
            pad_mask = gene_ids.eq(pad_id)
            delta_pred, _ = model(gene_ids, values, pad_mask, pert_ids)
            for p, d in zip(delta_pred.cpu().numpy(), delta.numpy()):
                r = np.corrcoef(p, d)[0, 1]
                if not np.isnan(r):
                    pearsons.append(r)
    if not pearsons:
        return 0.0, 0.0
    return float(np.mean(pearsons)), float(np.std(pearsons))


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(LOG_PATH, "w") as logf:
        def log(msg):
            print(msg); logf.write(msg + "\n"); logf.flush()

        log("=" * 65)
        log("scGPT_brain(frozen) + HCE → Norman 섭동 예측")
        log("=" * 65)

        # [1] scGPT 로드
        log("\n[1] scGPT_brain 로드...")
        scgpt_model, vocab, args = load_scgpt()
        d_model = args["embsize"]
        pad_id  = vocab["<pad>"]
        for p in scgpt_model.parameters():
            p.requires_grad_(False)
        scgpt_model.eval()
        log(f"  vocab={len(vocab)}, d_model={d_model}, layers={args['nlayers']}")

        # [2] GO 온톨로지
        log("\n[2] MSigDB Hallmark 온톨로지...")
        from HCE.msigdb_ontology import build_hallmark_ontology
        dag, term_to_idx, pathway_genes = build_hallmark_ontology()
        n_go = len(term_to_idx)
        log(f"  GO terms: {n_go}, DAG nodes: {len(dag.nodes)}")

        # [3] Norman 데이터
        log("\n[3] Norman PertData 로드...")
        from gears import PertData
        pert_data = PertData(GEARS_DIR)
        pert_data.load(data_name="norman")
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=BATCH, test_batch_size=BATCH)
        adata = pert_data.adata
        log(f"  adata: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
        log(f"  train conds: {len(pert_data.set2conditions['train'])}, "
            f"test conds: {len(pert_data.set2conditions.get('test', []))}")

        # [4] 데이터셋
        log("\n[4] 데이터셋 변환 (scGPT 형식)...")
        dataset = NormanScGPTDataset(adata, vocab, pathway_genes, term_to_idx)
        n_genes = dataset.n_genes

        train_ds, val_ds, test_ds = get_gene_ood_splits(
            dataset, pert_data.set2conditions)
        log(f"  split → train:{len(train_ds)}, val:{len(val_ds)}, test:{len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2)

        # [5] 모델 & Loss
        log("\n[5] 모델 초기화...")
        model = ScGPTNormanPredictor(scgpt_model, n_genes, n_go, d_model).to(DEVICE)
        from HCE.loss import HierarchicalPerturbationLoss
        loss_fn = HierarchicalPerturbationLoss(
            ontology=dag, go_term_to_idx=term_to_idx,
            lambda_reg=1.0, lambda_cls=LAMBDA_HCE,
        ).to(DEVICE)

        trainable = list(model.predictor.parameters()) + list(model.go_head.parameters())
        opt = optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

        n_params = sum(p.numel() for p in trainable)
        log(f"  학습 파라미터: {n_params:,}  (scGPT {d_model}D frozen)")
        log(f"  λ_HCE={LAMBDA_HCE}, epochs={EPOCHS}, batch={BATCH}, device={DEVICE}")

        # [6] 학습
        log(f"\n[6] 학습 시작...")
        best_val, best_ep = -1.0, 0
        history = []
        t0 = time.time()

        for ep in range(1, EPOCHS + 1):
            model.train()
            total_loss, n_batch = 0.0, 0
            for gene_ids, values, pert_ids, delta, go_lbl in train_loader:
                gene_ids = gene_ids.to(DEVICE)
                values   = values.to(DEVICE)
                pert_ids = pert_ids.to(DEVICE)
                delta    = delta.to(DEVICE)
                go_lbl   = go_lbl.to(DEVICE)
                pad_mask = gene_ids.eq(pad_id)

                delta_pred, go_logits = model(gene_ids, values, pad_mask, pert_ids)
                loss, info = loss_fn(delta_pred, delta, go_logits, go_lbl)

                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); n_batch += 1

            scheduler.step()

            if ep % 3 == 0 or ep == EPOCHS:
                val_r, val_std = evaluate(model, val_loader, pad_id)
                elapsed = time.time() - t0
                log(f"  Ep {ep:2d} | loss={total_loss/n_batch:.4f} "
                    f"| val_pearson={val_r:.4f}±{val_std:.3f} ({elapsed:.0f}s)")
                history.append({"ep": ep, "loss": total_loss/n_batch,
                                "val_pearson": val_r})

                if val_r > best_val:
                    best_val = val_r; best_ep = ep

        # 최종 test 평가
        test_r, test_std = evaluate(model, test_loader, pad_id)

        # [7] 결과 요약
        log("\n" + "=" * 65)
        log("결과 비교")
        log("-" * 65)
        log(f"  {'모델':30s} {'Best Pearson':>13s} {'ep15 Pearson':>13s}")
        log(f"  {'GEARS baseline':30s} {'0.692':>13s} {'0.005':>13s}  ← 붕괴")
        log(f"  {'GEARS + HCE (λ=0.3)':30s} {'0.817':>13s} {'0.700':>13s}  ← 안정")
        log(f"  {f'scGPT_brain + HCE (λ={LAMBDA_HCE})':30s} "
            f"{best_val:>13.4f} {history[-1]['val_pearson']:>13.4f}")
        log(f"\n  → test Pearson = {test_r:.4f} ± {test_std:.4f}  (best ep={best_ep})")
        log("=" * 65)

        # 저장
        results = {
            "model": f"scGPT_brain + HCE (λ={LAMBDA_HCE})",
            "lambda_hce": LAMBDA_HCE,
            "epochs": EPOCHS,
            "d_model": d_model,
            "n_go_terms": n_go,
            "best_val_pearson": best_val,
            "best_epoch": best_ep,
            "test_pearson": test_r,
            "test_pearson_std": test_std,
            "history": history,
            "comparison": {
                "GEARS_baseline_best": 0.692,
                "GEARS_baseline_ep15": 0.005,
                "GEARS_HCE_best":      0.817,
                "GEARS_HCE_ep15":      0.700,
            },
        }
        with open(RESULT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        log(f"\n결과 저장: {RESULT_PATH}")
        log(f"로그:      {LOG_PATH}")


if __name__ == "__main__":
    main()
