from __future__ import annotations  


import argparse, sys, os, glob, math, re
import pickle
from pathlib import Path
from statistics import median
from typing import List, Tuple, Dict

# --- asegurar import de 'src' si se ejecuta como script ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import networkx as nx

from src.data.graph_builder import load_passes, passes_to_digraph

def save_graph_pickle(G, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def p95(values: List[float]) -> float:
    if not values: 
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s)-1, int(math.ceil(0.95*(len(s)-1)))))
    return float(s[k])

def sanitize_name(p: Path) -> str:
    base = p.stem
    base = re.sub(r"[^\w\-]+", "_", base)
    return base

def edge_avg_dist(data: Dict) -> float:
    w = data.get("weight", 0)
    dist_sum = float(data.get("dist_sum", 0.0))
    return (dist_sum / w) if w else 0.0

def edge_success_ratio(data: Dict) -> float:
    w = data.get("weight", 0)
    succ = data.get("success_count", 0)
    return (float(succ) / w) if w else 0.0

def top_pairs_str(G: nx.DiGraph, k: int = 3) -> str:
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append((d.get("weight", 0), u, v, edge_success_ratio(d)))
    edges.sort(reverse=True, key=lambda t: t[0])
    parts = []
    for w, u, v, pr in edges[:k]:
        parts.append(f"{u}->{v} w={int(w)} p={pr:.2f}")
    return "; ".join(parts)

def graph_dist_stats(G: nx.DiGraph) -> Tuple[float, float, float]:
    vals = [edge_avg_dist(d) for _, _, d in G.edges(data=True)]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return (float(sum(vals)/len(vals)), float(median(vals)), float(p95(vals)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="Carpeta con CSV de eventos (raw)")
    ap.add_argument("--outdir", required=True, help="Carpeta de salida (gpickle + reporte)")
    ap.add_argument("--near-dist", type=float, default=7.5, help="Distancia (m) para fallback espacial")
    ap.add_argument("--lookahead", type=int, default=10, help="Filas a mirar hacia adelante en fallback")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(glob.glob(str(raw_dir / "*.csv")))
    assert csvs, f"No se encontraron CSV en {raw_dir}"

    rows = []
    for f in csvs:
        csv_path = Path(f)
        print(f"[INFO] Procesando: {csv_path.name}")

        # 1) cargar pases e inferir receptores
        df = load_passes(csv_path, cm=None, near_dist=args.near_dist, lookahead_rows=args.lookahead)

        # 2) construir grafos (por match si existe; si no, GLOBAL)
        graphs = passes_to_digraph(df)

        # usualmente 1 grafo por CSV; si hay más, iteramos igual
        for mid, G in graphs.items():
            # 3) métricas
            nodes = G.number_of_nodes()
            edges = G.number_of_edges()
            intra_team_pct = 100.0  # por el filtrado intra-equipo del builder
            coverage_pct = 100.0 * (edges / max(1, edges))  # marcador (compat con tu antiguo formato)
            # si guardaste team_coverage_pct en graph attr, úsalo como coverage real de team labels
            if "team_coverage_pct" in G.graph:
                coverage_pct = float(G.graph["team_coverage_pct"])

            dist_mean, dist_median, dist_p95 = graph_dist_stats(G)
            tops = top_pairs_str(G, k=3)

            # 4) guardar grafo
            gname = f"passes_{sanitize_name(csv_path)}.gpickle"
            gpath = outdir / gname
            save_graph_pickle(G, gpath)

            rows.append({
                "csv": csv_path.name,
                "nodes": nodes,
                "edges": edges,
                "coverage_pct": round(coverage_pct, 2),
                "intra_team_pct": round(intra_team_pct, 2),
                "dist_mean": round(dist_mean, 2) if not math.isnan(dist_mean) else "",
                "dist_median": round(dist_median, 2) if not math.isnan(dist_median) else "",
                "dist_p95": round(dist_p95, 2) if not math.isnan(dist_p95) else "",
                "top_pairs": tops,
                "nx_path": str(gpath)
            })

    # 5) reporte
    report_path = outdir / "report_enhanced.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8")
    print(f"[OK] Guardado: {report_path.as_posix()}")

if __name__ == "__main__":
    main()