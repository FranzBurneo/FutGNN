from __future__ import annotations
import argparse, json, random, numpy as np, torch
from src.data.datasets import SnapshotLPDataset
from src.baselines.rankers import (
    score_indegree, score_uv_frequency, score_common_neighbors, score_random_team
)

RANKERS = {
    "indegree": score_indegree,
    "uvfreq":   score_uv_frequency,
    "comm":     score_common_neighbors,
    "random":   score_random_team,
}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def hits_at_k_for_snapshot(pyg, future_edge_index, ranker_fn, k=5):
    # evalúa todos los futuros del snapshot
    # future_edge_index: [2, P] positivos (u->v) en t+1
    if future_edge_index is None or future_edge_index.numel() == 0:
        return []
    hits = []
    for j in range(future_edge_index.size(1)):
        u = int(future_edge_index[0, j].item())
        v = int(future_edge_index[1, j].item())

        scores = ranker_fn(pyg, u)            # [(v, score), ...] solo intra-equipo
        # desempates reproducibles:
        scores = sorted(scores, key=lambda x: (-x[1], x[0]))
        topk = [vv for vv, _ in scores[:k]]
        hits.append(1.0 if v in topk else 0.0)
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--ranker", choices=RANKERS.keys(), required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    ranker_fn = RANKERS[args.ranker]

    split = json.load(open(args.split_file, "r", encoding="utf-8"))
    test_files = split["test"]

    ds = SnapshotLPDataset(test_files, step=args.step, negative_k=5)  # neg_k no afecta baseline
    hits = []
    for i in range(len(ds)):
        pyg, pos, neg, mask, meta = ds[i]
        fut = meta.get("future_edge_index")
        hits.extend(hits_at_k_for_snapshot(pyg, fut, ranker_fn, k=args.k))

    h5 = float(np.mean(hits)) if hits else 0.0
    print(f"[EVAL] Split={args.split_file} | BASELINE={args.ranker} | Hits@{args.k}(TEST)={h5:.3f}")

if __name__ == "__main__":
    main()