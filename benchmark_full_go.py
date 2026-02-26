"""
benchmark_full_go.py
--------------------
Full GO (BP+CC+MF, ~3,693 terms) 온톨로지로 K562 OOD 벤치마크.
gene2go 실제 어노테이션 기반 GO 라벨 사용.

실행:
    python -m HCE.benchmark_full_go
"""
from __future__ import annotations
import sys, os, json, pickle, time
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score
import scipy.sparse

import HCE.config as cfg

DATA_PATH   = cfg.K562_DATA
GO_CACHE    = cfg.GO_CACHE
GENE2GO     = cfg.GENE2GO
RESULT_PATH = os.path.join(cfg.RESULTS_ROOT, "benchmark_full_go.json")
LOG_PATH    = os.path.join(cfg.RESULTS_ROOT, "benchmark_full_go.log")

LAMBDA_SWEEP = [0.0, 0.05, 0.1]
N_GENES = 2000
EPOCHS  = 20
BATCH   = 256
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ── GO 라벨 생성 (gene2go 기반) ─────────────────────────────────────
def build_gene_go_labels(pert_genes, term_to_idx, dag, gene2go):
    """
    섭동 유전자 → GO term 라벨 (gene2go 어노테이션 직접 사용).
    섭동 유전자의 직접 GO 어노테이션 + 조상 전파.
    """
    B = len(pert_genes)
    n_go = len(term_to_idx)
    labels = torch.zeros(B, n_go)

    for i, gene in enumerate(pert_genes):
        if gene not in gene2go:
            continue
        for go_term in gene2go[gene]:
            # 직접 어노테이션된 term + 모든 조상
            ancestors = dag.get_ancestors(go_term, include_self=True)
            for anc in ancestors:
                if anc in term_to_idx:
                    labels[i, term_to_idx[anc]] = 1.0
    return labels


# ── K562 데이터셋 (Full GO 라벨) ─────────────────────────────────────
class K562FullGODataset(Dataset):
    def __init__(self, adata_path, term_to_idx, dag, gene2go, gene_subset=2000):
        import scanpy as sc
        print(f"[K562FullGO] 로딩: {adata_path}")
        adata = sc.read_h5ad(adata_path)

        # 컨트롤
        ctrl_mask = adata.obs.get("core_control", None)
        if ctrl_mask is not None:
            ctrl_idx = np.where(ctrl_mask.values)[0]
        else:
            ctrl_idx = np.where(adata.obs_names.str.contains("non-targeting"))[0]

        X = adata.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        # 분산 상위 유전자
        var = X.var(axis=0)
        top_idx = np.argsort(var)[-gene_subset:]
        X = X[:, top_idx]
        gene_names = adata.var_names[top_idx].tolist()

        self.ctrl_mean = X[ctrl_idx].mean(axis=0)
        self.delta = X - self.ctrl_mean

        # 섭동 유전자 파싱
        def parse_gene(obs_name):
            parts = obs_name.split("_")
            if len(parts) >= 2:
                return parts[1]
            return ""

        self.pert_genes = [parse_gene(n) for n in adata.obs_names]

        # var_names는 Ensembl ID이므로 gene_name(symbol) 컬럼으로 pert_mask 생성
        if 'gene_name' in adata.var.columns:
            sym_at_top = set(adata.var['gene_name'].values[top_idx].tolist())
        else:
            sym_at_top = set(gene_names)

        # 섭동 마스크
        self.pert_mask = np.zeros(len(adata), dtype=np.float32)
        for j, g in enumerate(self.pert_genes):
            if g in sym_at_top:
                self.pert_mask[j] = 1.0

        # GO 라벨 (gene2go 기반)
        print(f"[K562FullGO] GO 라벨 생성 중 (n={len(term_to_idx)} terms)...")
        self.go_labels = build_gene_go_labels(
            self.pert_genes, term_to_idx, dag, gene2go
        )
        pos_rate = self.go_labels.mean().item()
        coverage = (self.go_labels.sum(1) > 0).float().mean().item()
        print(f"  GO 라벨 양성 비율: {pos_rate:.3f}, 유전자 커버리지: {coverage:.3f}")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.delta = torch.tensor(self.delta, dtype=torch.float32)
        self.n_genes = gene_subset
        self.gene_names = gene_names
        print(f"  완료: {len(self.X)}개 샘플, {gene_subset}개 유전자")

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.delta[idx],
            self.go_labels[idx],
            self.pert_mask[idx],
        )

    def get_splits(self, val_ratio=0.1, test_ratio=0.1, seed=42):
        rng = np.random.default_rng(seed)
        n = len(self)
        # Gene-OOD split: 측정된 샘플만
        measured = np.where(self.pert_mask > 0)[0]
        unique_genes = list(set(self.pert_genes[i] for i in measured))
        rng.shuffle(unique_genes)
        n_ood = max(1, int(len(unique_genes) * 0.2))
        ood_genes = set(unique_genes[:n_ood])
        id_measured = [i for i in measured if self.pert_genes[i] not in ood_genes]
        ood_test = [i for i in measured if self.pert_genes[i] in ood_genes]
        # 나머지 (미측정)
        unmeasured = np.where(self.pert_mask == 0)[0].tolist()
        all_train = unmeasured + id_measured
        rng.shuffle(all_train)
        n_val = int(len(all_train) * val_ratio)
        val_idx = all_train[:n_val]
        train_idx = all_train[n_val:]
        id_test = id_measured[:min(500, len(id_measured))]
        return (Subset(self, train_idx), Subset(self, val_idx),
                Subset(self, id_test), Subset(self, ood_test))


# ── 간단한 MLP 모델 (Full GO head) ────────────────────────────────────
class FullGOPredictor(nn.Module):
    def __init__(self, n_genes, n_go, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes * 2, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.expr_head = nn.Linear(hidden, n_genes)
        self.go_head   = nn.Linear(hidden, n_go)

    def forward(self, expr, pert_mask):
        x = torch.cat([expr, pert_mask.unsqueeze(1).expand_as(expr)], dim=1)
        h = self.encoder(x)
        return self.expr_head(h), self.go_head(h)


def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n"); f.flush()


def run_benchmark(dataset, term_to_idx, dag, lam, logf):
    from HCE.loss import HierarchicalPerturbationLoss

    train_ds, val_ds, id_ds, ood_ds = dataset.get_splits()
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    model = FullGOPredictor(N_GENES, len(term_to_idx)).to(DEVICE)
    loss_fn = HierarchicalPerturbationLoss(
        ontology=dag, go_term_to_idx=term_to_idx,
        lambda_reg=1.0, lambda_cls=lam,
    ).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for expr, delta, go_lbl, pmask in train_loader:
            expr   = expr.to(DEVICE)
            delta  = delta.to(DEVICE)
            go_lbl = go_lbl.to(DEVICE)
            pmask  = pmask.to(DEVICE)
            pred, go_logits = model(expr, pmask)
            loss, info = loss_fn(pred, delta, go_logits, go_lbl)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            log(f"  [λ={lam}] ep{ep:2d} reg={info['loss_regression']:.4f} "
                f"hier={info['loss_hier_cls']:.4f}", logf)

    def evaluate(subset):
        model.eval()
        loader = DataLoader(subset, batch_size=512)
        pearsons, go_preds, go_trues = [], [], []
        with torch.no_grad():
            for expr, delta, go_lbl, pmask in loader:
                expr = expr.to(DEVICE); pmask = pmask.to(DEVICE)
                pred, go_logits = model(expr, pmask)
                for p, d in zip(pred.cpu().numpy(), delta.numpy()):
                    r = np.corrcoef(p, d)[0, 1]
                    if not np.isnan(r): pearsons.append(r)
                go_preds.append(torch.sigmoid(go_logits).cpu().numpy())
                go_trues.append(go_lbl.numpy())
        if not go_preds:
            return {"pearson": 0.0, "pearson_std": 0.0, "go_auroc": 0.0, "n_valid_terms": 0}
        go_preds = np.concatenate(go_preds)
        go_trues = np.concatenate(go_trues)
        # AUROC (valid terms only)
        aurocs = []
        for k in range(go_trues.shape[1]):
            if go_trues[:, k].sum() > 0 and go_trues[:, k].sum() < len(go_trues):
                try:
                    aurocs.append(roc_auc_score(go_trues[:, k], go_preds[:, k]))
                except Exception:
                    pass
        return {
            "pearson": float(np.mean(pearsons)),
            "pearson_std": float(np.std(pearsons)),
            "go_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
            "n_valid_terms": len(aurocs),
        }

    elapsed = time.time() - t0
    id_res  = evaluate(id_ds)
    ood_res = evaluate(ood_ds)
    log(f"  [λ={lam}] 완료 ({elapsed:.1f}s)", logf)
    log(f"  [λ={lam}] ID  | Pearson={id_res['pearson']:.4f} "
        f"GO_AUROC={id_res['go_auroc']:.4f} "
        f"valid_terms={id_res['n_valid_terms']}", logf)
    log(f"  [λ={lam}] OOD | Pearson={ood_res['pearson']:.4f} "
        f"GO_AUROC={ood_res['go_auroc']:.4f}", logf)
    return {"id": id_res, "ood": ood_res}


def main():
    from HCE.go_ontology_full import load_or_build_go_dag

    with open(LOG_PATH, "w") as logf:
        log("=" * 65, logf)
        log("Full GO Benchmark (BP+CC+MF, gene2go 라벨)", logf)
        log("=" * 65, logf)

        # GO DAG 로드
        log("\n[1] Full GO DAG 로드...", logf)
        dag, term_to_idx, _ = load_or_build_go_dag(
            min_genes=50,
            max_genes=2000,
        )
        log(f"  GO terms: {len(term_to_idx)}, DAG nodes: {len(dag.nodes)}", logf)

        # gene2go 로드
        log("\n[2] gene2go 로드...", logf)
        with open(GENE2GO, "rb") as f:
            gene2go_raw = pickle.load(f)
        # gene2go는 {entrez_id: set(go_terms)} 또는 {gene_symbol: set(go_terms)}
        # 형식 확인
        sample_key = next(iter(gene2go_raw))
        log(f"  gene2go 형식 예시: {sample_key} → {list(gene2go_raw[sample_key])[:3]}", logf)
        gene2go = gene2go_raw

        # 데이터셋
        log("\n[3] K562 데이터셋 준비...", logf)
        dataset = K562FullGODataset(DATA_PATH, term_to_idx, dag, gene2go, N_GENES)

        # 벤치마크
        results = {}
        log("\n" + "─" * 65, logf)
        for lam in LAMBDA_SWEEP:
            log(f"\n  모델: Full GO λ={lam}", logf)
            results[f"λ={lam}"] = run_benchmark(dataset, term_to_idx, dag, lam, logf)

        # 저장
        log("\n" + "=" * 65, logf)
        log(f"{'모델':12s} {'분할':6s} {'Pearson':>8s} {'GO_AUROC':>10s} {'valid_terms':>12s}", logf)
        log("-" * 65, logf)
        for lam_key, res in results.items():
            for split, r in res.items():
                marker = " ← OOD" if split == "ood" else ""
                log(f"{lam_key:12s} {split:6s} {r['pearson']:8.4f} "
                    f"{r['go_auroc']:10.4f} {r.get('n_valid_terms', 0):12d}{marker}", logf)

        with open(RESULT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        log(f"\n결과 저장: {RESULT_PATH}", logf)


if __name__ == "__main__":
    main()
