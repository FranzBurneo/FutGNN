# scripts/eval_by_match.py
import argparse, glob, torch
from torch.utils.data import DataLoader
from src.data.datasets import SnapshotLPDataset
from src.models.gnn_lp import GNNSimpleLP
from src.train.train_lp import eval_hits_at_k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/raw/*.csv")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--neg", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    ckpt = torch.load(args.ckpt, map_location=args.device)
    margs = ckpt["args"]
    model = GNNSimpleLP(in_dim=2, hidden=margs["hidden"], layers=margs["layers"], out=margs["out"]).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()

    for f in files:
        ds = SnapshotLPDataset([f], step=args.step, negative_k=args.neg)
        dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=lambda b: list(zip(*b)))
        h5 = eval_hits_at_k(model, dl, K=5, device=args.device)
        print(f"{f} -> Hits@5={h5:.3f}")

if __name__ == "__main__":
    main()