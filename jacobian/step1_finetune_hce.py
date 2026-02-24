"""
step1_finetune_hce.py
---------------------
scGPT_brain (13.2M brain cells) + HCE Cell Type Classifier 파인튜닝.
Brain Cell Ontology (4 계층) + HierarchicalCrossEntropyLoss.

실행:
    python -m HCE.jacobian.step1_finetune_hce
"""
from __future__ import annotations
import sys, os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scanpy as sc

import HCE.config as cfg

SCGPT_DIR  = cfg.SCGPT_BRAIN_DIR
DATA_PATH  = cfg.BRAIN_ATLAS
SAVE_DIR   = cfg.JACOBIAN_RESULTS
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_CELLS    = 500     # 세포 유형별 샘플 수
MAX_SEQ    = 1200    # 유전자 토큰 수
N_BINS     = 51
EPOCHS     = 15
BATCH      = 32

CELL_TYPES  = ["RG", "Neuroblast", "Ext", "Inh"]
LEAF_TO_IDX = {"RG": 0, "Neuroblast": 1, "Ext": 2, "Inh": 3}

# ── Brain Cell Ontology ──────────────────────────────────────────────
def build_brain_cell_ontology():
    """
    Cell
    └── Neural Cell
        ├── Neuron
        │   ├── Excitatory Neuron  → Ext
        │   └── Inhibitory Neuron  → Inh
        └── Neural Progenitor
            ├── Radial Glia        → RG
            └── Neuroblast         → Neuroblast
    """
    from HCE.ontology import OntologyDAG
    dag = OntologyDAG()
    dag.add_edge("Excitatory_Neuron",  "Neuron")
    dag.add_edge("Inhibitory_Neuron",  "Neuron")
    dag.add_edge("Radial_Glia",        "Neural_Progenitor")
    dag.add_edge("Neuroblast_cell",    "Neural_Progenitor")
    dag.add_edge("Neuron",             "Neural_Cell")
    dag.add_edge("Neural_Progenitor",  "Neural_Cell")
    dag.add_edge("Neural_Cell",        "Cell")

    term_to_idx = {
        "Radial_Glia":       LEAF_TO_IDX["RG"],
        "Neuroblast_cell":   LEAF_TO_IDX["Neuroblast"],
        "Excitatory_Neuron": LEAF_TO_IDX["Ext"],
        "Inhibitory_Neuron": LEAF_TO_IDX["Inh"],
    }
    return dag, term_to_idx


# ── scGPT_brain 로드 ─────────────────────────────────────────────────
def load_scgpt_brain():
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab

    vocab = GeneVocab.from_file(os.path.join(SCGPT_DIR, "vocab.json"))
    with open(os.path.join(SCGPT_DIR, "args.json")) as f:
        args = json.load(f)

    pad_id = vocab[args["pad_token"]]
    model  = TransformerModel(
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
    ckpt = torch.load(os.path.join(SCGPT_DIR, "best_model.pt"),
                      map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    print(f"  scGPT_brain 로드 완료 (vocab={len(vocab)}, "
          f"d={args['embsize']}, layers={args['nlayers']})")
    return model, vocab, args


# ── 데이터셋: fetal brain → scGPT 입력 형식 ─────────────────────────
class BrainCellDataset(Dataset):
    """
    fetal brain h5ad → (gene_ids, values, label) scGPT 형식.
    """
    def __init__(self, adata, vocab, n_cells_per_type=N_CELLS,
                 max_seq=MAX_SEQ, n_bins=N_BINS, seed=42):
        rng = np.random.default_rng(seed)
        pad_id = vocab["<pad>"]

        # 세포 유형별 샘플링
        samples, labels_raw = [], []
        for ct, lb in LEAF_TO_IDX.items():
            mask = adata.obs["Cell Type"] == ct
            idx  = np.where(mask)[0]
            n    = min(n_cells_per_type, len(idx))
            chosen = rng.choice(idx, n, replace=False)
            samples.extend(chosen); labels_raw.extend([lb] * n)
        print(f"  총 {len(samples)}개 세포 샘플 (유형별 최대 {n_cells_per_type})")

        # 겹치는 유전자 (adata var_names ∩ vocab)
        gene_names = adata.var_names.tolist()
        common = [(i, gene_names[i]) for i in range(len(gene_names))
                  if gene_names[i] in vocab]
        print(f"  adata-vocab 겹치는 유전자: {len(common)}/{len(gene_names)}")

        gene_col_idx   = [c[0] for c in common]
        gene_token_ids = [vocab[c[1]] for c in common]

        # 발현값 준비
        X = adata.layers["logcounts"]
        if hasattr(X, "toarray"): X = X.toarray()
        X = X.astype(np.float32)

        self.gene_ids_list = []
        self.values_list   = []
        self.labels        = torch.tensor(labels_raw, dtype=torch.long)

        for cell_i in samples:
            expr = X[cell_i, gene_col_idx]  # (n_common,)
            # 발현값 top-max_seq 유전자 선택
            n_sel = min(max_seq, len(expr))
            top_k = np.argsort(expr)[-n_sel:][::-1]
            sel_gene_ids  = np.array(gene_token_ids)[top_k]
            sel_vals      = expr[top_k]

            # 정규화 & binning
            sel_vals = np.log1p(sel_vals)
            max_v    = sel_vals.max() + 1e-6
            sel_vals = sel_vals / max_v  # [0, 1]
            binned   = np.floor(sel_vals * (n_bins - 1)).astype(np.float32)

            # 패딩
            pad_len = max_seq - len(sel_gene_ids)
            g_pad   = np.full(pad_len, pad_id, dtype=np.int64)
            v_pad   = np.full(pad_len, -2.0, dtype=np.float32)  # pad_value
            gene_ids_padded = np.concatenate([sel_gene_ids, g_pad])
            vals_padded     = np.concatenate([binned, v_pad])

            self.gene_ids_list.append(gene_ids_padded)
            self.values_list.append(vals_padded)

        self.gene_ids = torch.tensor(np.stack(self.gene_ids_list), dtype=torch.long)
        self.values   = torch.tensor(np.stack(self.values_list),   dtype=torch.float32)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.gene_ids[idx], self.values[idx], self.labels[idx]


# ── HCE 분류 헤드 ─────────────────────────────────────────────────────
class ScGPTBrainHCE(nn.Module):
    """
    scGPT_brain + HierarchicalCrossEntropyLoss 헤드.
    scGPT frozen, HCE head만 학습.
    """
    def __init__(self, scgpt_model, n_classes, d_model=512):
        super().__init__()
        self.scgpt  = scgpt_model
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

    def get_cell_emb(self, gene_ids, values, src_key_padding_mask):
        """
        scGPT forward → cell embedding (mean pooling, no CLS).
        requires_grad=True on values → gradient 흐름 유지.
        """
        output = self.scgpt(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            CLS=False,
            MVC=False,
            ECS=False,
        )
        # mlm_output: (B, seq_len, d_model) → mean over non-pad tokens
        mlm = output["mlm_output"]  # (B, L, D)
        pad_mask = src_key_padding_mask  # (B, L), True=pad
        # non-pad 위치 평균
        non_pad = (~pad_mask).float().unsqueeze(-1)  # (B, L, 1)
        cell_emb = (mlm * non_pad).sum(1) / non_pad.sum(1).clamp(min=1)  # (B, D)
        return cell_emb

    def forward(self, gene_ids, values, src_key_padding_mask):
        cell_emb = self.get_cell_emb(gene_ids, values, src_key_padding_mask)
        logits   = self.cls_head(cell_emb)  # (B, n_classes)
        return logits, cell_emb


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_path = os.path.join(SAVE_DIR, "step1_finetune.log")

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log("Step 1: scGPT_brain + HCE Brain Cell Ontology 파인튜닝")
    log("=" * 60)

    # ── 온톨로지 ──────────────────────────────────────────────────────
    log("\n[1] Brain Cell Ontology 구성...")
    dag, term_to_idx = build_brain_cell_ontology()
    log(f"  DAG: {len(dag.nodes)} nodes, {sum(len(v) for v in dag.parents.values())} edges")
    log(f"  리프: {list(term_to_idx.keys())}")

    # ── 모델 로드 ──────────────────────────────────────────────────────
    log("\n[2] scGPT_brain 로드...")
    scgpt_model, vocab, args = load_scgpt_brain()

    # scGPT frozen
    for p in scgpt_model.parameters():
        p.requires_grad_(False)
    log("  scGPT frozen (HCE head만 학습)")

    model = ScGPTBrainHCE(scgpt_model, n_classes=len(term_to_idx), d_model=args["embsize"])
    model = model.to(DEVICE)

    # ── 데이터 ────────────────────────────────────────────────────────
    log("\n[3] Fetal brain 데이터 준비...")
    adata = sc.read_h5ad(DATA_PATH)
    dataset = BrainCellDataset(adata, vocab, n_cells_per_type=N_CELLS)

    n = len(dataset)
    n_val = int(n * 0.1)
    indices = torch.randperm(n).tolist()
    train_ds = torch.utils.data.Subset(dataset, indices[n_val:])
    val_ds   = torch.utils.data.Subset(dataset, indices[:n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH)
    log(f"  train: {len(train_ds)}, val: {len(val_ds)}")

    # ── HCE Loss ──────────────────────────────────────────────────────
    from HCE.loss import HierarchicalCrossEntropyLoss
    criterion = HierarchicalCrossEntropyLoss(
        ontology=dag,
        term_to_idx=term_to_idx,
        alpha=0.7,
        monotone_coef=0.1,
    ).to(DEVICE)

    opt = optim.AdamW(model.cls_head.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    # ── 학습 ──────────────────────────────────────────────────────────
    log(f"\n[4] 학습 (epochs={EPOCHS}, batch={BATCH}, device={DEVICE})")
    best_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for gene_ids, values, labels in train_loader:
            gene_ids = gene_ids.to(DEVICE)
            values   = values.to(DEVICE)
            labels   = labels.to(DEVICE)
            pad_mask = gene_ids.eq(vocab["<pad>"])
            logits, _ = model(gene_ids, values, pad_mask)
            loss, _   = criterion(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for gene_ids, values, labels in val_loader:
                gene_ids = gene_ids.to(DEVICE)
                values   = values.to(DEVICE)
                labels   = labels.to(DEVICE)
                pad_mask = gene_ids.eq(vocab["<pad>"])
                logits, _ = model(gene_ids, values, pad_mask)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total   += len(labels)
        acc = correct / max(total, 1)
        scheduler.step()

        if ep % 3 == 0 or ep == EPOCHS:
            log(f"  Ep {ep:2d} | loss={total_loss/len(train_loader):.4f} "
                f"| val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            ckpt_path = os.path.join(SAVE_DIR, "hce_brain_best.pt")
            torch.save({
                "epoch": ep,
                "model_state": model.cls_head.state_dict(),
                "val_acc": acc,
                "dag_nodes": list(dag.nodes.keys()),
                "term_to_idx": term_to_idx,
            }, ckpt_path)

    log(f"\n  최고 val_acc: {best_acc:.4f}")
    log(f"  저장: {SAVE_DIR}/hce_brain_best.pt")

    # ── 세포 유형별 정확도 ────────────────────────────────────────────
    log("\n[5] 세포 유형별 분류 정확도")
    model.eval()
    per_type = {ct: [0, 0] for ct in CELL_TYPES}
    idx_to_ct = {v: k for k, v in LEAF_TO_IDX.items()}
    with torch.no_grad():
        for gene_ids, values, labels in val_loader:
            gene_ids = gene_ids.to(DEVICE); values = values.to(DEVICE)
            pad_mask = gene_ids.eq(vocab["<pad>"])
            logits, _ = model(gene_ids, values, pad_mask)
            preds = logits.argmax(1).cpu()
            for lb, pr in zip(labels, preds):
                ct = idx_to_ct[lb.item()]
                per_type[ct][1] += 1
                if lb == pr: per_type[ct][0] += 1
    for ct, (c, t) in per_type.items():
        log(f"  {ct:12s}: {c}/{t} ({c/max(t,1):.3f})")

    log("\nStep 1 완료. Step 2 (Jacobian 계산)을 실행하세요.")
    log(f"  python -m HCE.jacobian.step2_hce_jacobian")


if __name__ == "__main__":
    main()
