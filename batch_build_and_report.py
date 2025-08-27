#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch builder & reporter for pass networks.
- Itera sobre data/raw/*.csv
- Construye grafos usando src.data.graph_builder (sin sobrescribir GLOBAL)
- Guarda passes_<stem>.gpickle / .pt y un report.csv con métricas
"""
from pathlib import Path
import argparse, sys, pickle
import pandas as pd
import torch
import networkx as nx

# Import from local package
sys.path.append(str(Path(__file__).resolve().parents[0]))
from src.data.graph_builder import load_passes, passes_to_digraph, nx_to_pyg

def save_nx_gpickle(G: nx.Graph, path: Path) -> None:
    try:
        from networkx.readwrite.gpickle import write_gpickle as write_gpickle_nx
        write_gpickle_nx(G, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw", help="Carpeta con CSVs")
    ap.add_argument("--outdir", default="data/processed", help="Carpeta de salida")
    ap.add_argument("--near-dist", type=float, default=10.0, help="Umbral de cercanía para inferir receptor")
    ap.add_argument("--lookahead", type=int, default=15, help="Filas siguientes a revisar para inferir receptor")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    csvs = sorted([p for p in raw_dir.glob("*.csv") if p.is_file()])
    if not csvs:
        print(f"[WARN] No se encontraron CSVs en {raw_dir.resolve()}")
        return

    for csv_path in csvs:
        stem = csv_path.stem.replace(" ", "_").replace("/", "_")
        print(f"[INFO] Procesando: {csv_path.name}")

        df = load_passes(csv_path, cm=None, near_dist=args.near_dist, lookahead_rows=args.lookahead)
        total = len(df)
        with_to = int(df["to_id"].notna().sum())
        coverage = (with_to / total) if total else 0.0

        # construir grafo (global)
        graphs = passes_to_digraph(df, by_match=None)
        G = graphs.get("GLOBAL")
        if G is None:
            print(f"[WARN] No se generó grafo para {csv_path.name}")
            continue

        # guardar con nombre único
        nx_path = outdir / f"passes_{stem}.gpickle"
        pt_path = outdir / f"passes_{stem}.pt"
        save_nx_gpickle(G, nx_path)
        torch.save(nx_to_pyg(G), pt_path)

        # resumen para reporte
        rows.append({
            "csv": csv_path.name,
            "passes_total": total,
            "passes_with_receiver": with_to,
            "coverage_pct": round(100*coverage, 2),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "nx_path": str(nx_path),
            "pt_path": str(pt_path),
        })

    if rows:
        rep = pd.DataFrame(rows).sort_values("edges", ascending=False)
        rep_path = outdir / "report.csv"
        rep.to_csv(rep_path, index=False)
        print(f"[OK] Reporte guardado en: {rep_path}")
    else:
        print("[WARN] No se generaron resultados.")

if __name__ == "__main__":
    main()
