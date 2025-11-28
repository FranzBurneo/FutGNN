# scripts/eval_on_test.py
from __future__ import annotations
import argparse, json, torch, glob
from torch.utils.data import DataLoader
from src.data.datasets import SnapshotLPDataset
from src.models.gnn_lp import GNNSimpleLP
from src.train.train_lp import eval_hits_at_k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    split = json.load(open(args.split_file, "r"))
    test_files = split.get("test", [])
    assert test_files, "No hay archivos en TEST en el split."

    ds_te = SnapshotLPDataset(test_files, step=args.step, negative_k=args.neg)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, collate_fn=lambda b: list(zip(*b)))

    ckpt = torch.load(args.ckpt, map_location=args.device)
    margs = ckpt.get("args", {})
    model = GNNSimpleLP(in_dim=2,
                        hidden=margs.get("hidden", 64),
                        layers=margs.get("layers", 2),
                        out=margs.get("out", 64)).to(args.device)
    model.load_state_dict(ckpt["model"])

    hits5 = eval_hits_at_k(model, dl_te, K=5, device=args.device)
    print(f"[TEST] Hits@5={hits5:.3f}  |  Snapshots={len(ds_te)}  |  Partidos={len(test_files)}")

if __name__ == "__main__":
    main()