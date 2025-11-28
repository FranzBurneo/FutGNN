from __future__ import annotations
import argparse, json, glob, os, torch
from torch.utils.data import DataLoader
from src.data.datasets import SnapshotLPDataset
from src.models.gnn_lp import GNNSimpleLP
from src.train.train_lp import eval_hits_at_k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.split_file, "r", encoding="utf-8") as f:
        split = json.load(f)
    test_files = split.get("test") or split.get("TEST") or []
    assert test_files, "No hay archivos en TEST."

    ds_te = SnapshotLPDataset(test_files, step=args.step, negative_k=args.neg)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, collate_fn=lambda b: list(zip(*b)))

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    h = ckpt.get("args", {})
    model = GNNSimpleLP(in_dim=2,
                        hidden=h.get("hidden", 64),
                        layers=h.get("layers", 2),
                        out=h.get("out", 64)).to(args.device)
    model.load_state_dict(ckpt["model"])

    hits5 = eval_hits_at_k(model, dl_te, K=5, device=args.device)
    print(f"[EVAL] Split={os.path.basename(args.split_file)} | CKPT={os.path.basename(args.ckpt)} | Hits@5(TEST)={hits5:.3f}")

if __name__ == "__main__":
    main()