# scripts/demo_predict.py
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from src.data.datasets import SnapshotLPDataset
from src.models.gnn_lp import GNNSimpleLP

def _norm_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_name_map_from_csv(csv_path: str) -> dict[int, str]:
    """
    Intenta mapear id_real -> nombre a partir del CSV.
    Busca columnas típicas (case-insensitive):
      id:  ['playerid','relatedplayerid','id_jugador','jugador_id']
      name:['shortname','playername','player','jugador','name']
    Devuelve dict; si no hay nombre, usa str(id).
    """
    df = pd.read_csv(csv_path)
    df = _norm_cols(df)

    # Candidatos de columnas
    id_cols = [c for c in ['playerid','relatedplayerid','id_jugador','jugador_id','player_id'] if c in df.columns]
    name_cols = [c for c in ['shorname','playername','player','jugador','name','player_name'] if c in df.columns]

    name_map = {}

    # Recolectar (id, nombre) donde tengamos ambas columnas
    if id_cols and name_cols:
        for idc in id_cols:
            for nc in name_cols:
                sub = df[[idc, nc]].dropna().drop_duplicates()
                for rid, nm in zip(sub[idc], sub[nc]):
                    try:
                        rid_i = int(rid)
                        nm_s = str(nm).strip()
                        if nm_s:
                            name_map[rid_i] = nm_s
                    except Exception:
                        pass
    # Si no hubo nombres, al menos aseguremos IDs observados
    if not name_map and id_cols:
        ids = pd.concat([df[c] for c in id_cols], ignore_index=True).dropna().unique().tolist()
        for rid in ids:
            try:
                name_map[int(rid)] = str(int(rid))
            except Exception:
                continue

    return name_map

def build_snapshot_graph(pyg):
    G = nx.DiGraph()
    n = pyg.num_nodes
    node_ids = getattr(pyg, "node_ids", None)
    for i in range(n):
        rid = int(node_ids[i].item()) if node_ids is not None and len(node_ids) == n else i
        G.add_node(i, real_id=rid)

    ei = pyg.edge_index
    if ei is not None and ei.numel() > 0:
        src = ei[0].tolist(); dst = ei[1].tolist()
        for u, v in zip(src, dst):
            G.add_edge(int(u), int(v))
    return G

@torch.no_grad()
def topk_for_one_future(model, pyg, u_idx: int, v_idx: int, k: int = 5):
    device = next(model.parameters()).device
    pyg = pyg.to(device)
    z = model.encode(pyg)

    all_nodes = torch.arange(pyg.num_nodes, device=device)
    cand_v = all_nodes[all_nodes != u_idx]
    pairs = torch.stack([torch.full_like(cand_v, u_idx), cand_v], dim=0)
    logits = model.score_pairs(z, pairs)
    probs = torch.sigmoid(logits)

    topk_probs, topk_idx = torch.topk(probs, k=min(k, probs.numel()))
    topk_v = cand_v[topk_idx]

    gt_pos = None
    if (cand_v == v_idx).any():
        idx = (cand_v == v_idx).nonzero(as_tuple=True)[0].item()
        score_gt = probs[idx].item()
        gt_pos = int((probs >= score_gt).sum().item())  # 1-based
    return topk_v.tolist(), topk_probs.tolist(), gt_pos

def pretty_print_table(u_idx, u_real, topV, topP, gt_v_idx, gt_v_real, gt_pos, k):
    print("\n=== Predicción de receptor ===")
    print(f"Pasador: u_idx={u_idx} (id_real={u_real})")
    print(f"Top-{k} candidatos:")
    print(f"{'Rank':>4}  {'v_idx':>5}  {'prob':>7}  {'GT':>3}  {'id_real':>10}")
    print("-" * 40)
    for r, (v, p) in enumerate(zip(topV, topP), start=1):
        is_gt = "✔" if v == gt_v_idx else ""
        rid = gt_v_real if v == gt_v_idx else ""
        print(f"{r:>4}  {v:>5}  {p:7.3f}  {is_gt:>3}  {str(rid):>10}")
    if gt_pos is not None and gt_pos <= k:
        print(f"\n✅ ACIERTO (GT en Top-{k}, posición {gt_pos}).")
    else:
        print(f"\n❌ FALLO (GT fuera del Top-{k}).")

def _label_for(n_idx: int, real_id: int, name_map: dict[int,str], mode: str) -> str:
    nm = name_map.get(real_id, str(real_id))
    if mode == "idx":
        return str(n_idx)
    if mode == "real":
        return str(real_id)
    if mode == "name":
        return nm
    if mode == "name+idx":
        return f"{nm}\n({n_idx})"
    return nm  # por defecto

def draw_snapshot(pyg, u_idx, v_idx_gt, topV, out_png, title="Demo predicción",
                  seed=7, name_map: dict[int,str] | None = None, label_mode: str = "name"):
    G = build_snapshot_graph(pyg)
    pos = nx.spring_layout(G, seed=seed)

    node_ids = getattr(pyg, "node_ids", None)
    labels = {}
    for n in G.nodes():
        real_id = int(node_ids[n].item()) if node_ids is not None else n
        if name_map is None:
            labels[n] = _label_for(n, real_id, {}, label_mode="idx")
        else:
            labels[n] = _label_for(n, real_id, name_map, label_mode)

    node_color = []
    top_set = set(topV)
    for n in G.nodes():
        if n == u_idx: node_color.append("#1f77b4")      # pasador (azul)
        elif n == v_idx_gt: node_color.append("#2ca02c") # GT (verde)
        elif n in top_set: node_color.append("#ff7f0e")  # candid. top-k (naranja)
        else: node_color.append("#cccccc")               # otros (gris)

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_color, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=True, arrowsize=12, edge_color="#999999", width=1.0)
    if G.has_node(u_idx) and G.has_node(v_idx_gt):
        nx.draw_networkx_edges(G, pos, edgelist=[(u_idx, v_idx_gt)],
                               edge_color="#d62728", width=2.8, arrows=True, arrowsize=14)

    plt.title(title); plt.axis("off")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"[OK] Figura guardada en: {out_png}")


def parse_args():
    ap = argparse.ArgumentParser(description="Demo de predicción de receptor (Link Prediction)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--glob", default="data/raw/*.csv")
    ap.add_argument("--match-like", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--idx", type=int, default=None, help="snapshot idx opcional")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-prefix", default="docs/figs/demo", help="prefijo para PNG/CSV de salida")
    ap.add_argument("--label-mode", choices=["idx","real","name","name+idx"], default="name",
                help="Texto mostrado en los nodos del grafo")
    return ap.parse_args()

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    h = ckpt.get("args", {}).get("hidden", 64)
    L = ckpt.get("args", {}).get("layers", 2)
    o = ckpt.get("args", {}).get("out", 64)
    model = GNNSimpleLP(in_dim=2, hidden=h, layers=L, out=o).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    files = sorted(glob.glob(args.glob))
    matches = [f for f in files if args.match_like.lower() in os.path.basename(f).lower()]
    assert matches, f"No se encontró partido que contenga: {args.match_like}"
    match = matches[0]

    ds = SnapshotLPDataset([match], step=args.step, negative_k=5)
    print(f"[INFO] Partido: {os.path.basename(match)} | snapshots={len(ds)}")

    idx = args.idx if args.idx is not None else max(0, len(ds)//10)
    pyg, pos, neg, mask, meta = ds[idx]

    fut = meta.get("future_edge_index")
    assert fut is not None and fut.numel() > 0, "Snapshot sin futuros disponibles."
    u_idx = int(fut[0, 0].item()); v_idx_gt = int(fut[1, 0].item())

    node_ids = getattr(pyg, "node_ids", None)
    u_real = int(node_ids[u_idx].item()) if node_ids is not None else u_idx
    v_real = int(node_ids[v_idx_gt].item()) if node_ids is not None else v_idx_gt

    name_map = load_name_map_from_csv(match)

    topV, topP, gt_pos = topk_for_one_future(model, pyg, u_idx, v_idx_gt, k=args.topk)
    pretty_print_table(u_idx, u_real, topV, topP, v_idx_gt, v_real, gt_pos, args.topk)

    safe = os.path.basename(match).replace(" ", "_").replace("/", "_")
    out_png = f"{args.out_prefix}_{safe}_t{meta['t_start']}-{meta['t_end']}.png"
    draw_snapshot(
        pyg, u_idx, v_idx_gt, topV, out_png,
        title=f"{os.path.basename(match)} | t={meta['t_start']}→{meta['t_end']}",
        name_map=name_map, label_mode=args.label_mode
    )

    out_csv = f"{Path(args.out_prefix).with_suffix('')}_{safe}_t{meta['t_start']}-{meta['t_end']}.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("rank,v_idx,prob,is_gt\n")
        for r, (v, p) in enumerate(zip(topV, topP), start=1):
            f.write(f"{r},{v},{p:.6f},{int(v == v_idx_gt)}\n")
    print(f"[OK] CSV guardado en: {out_csv}")

if __name__ == "__main__":
    main()