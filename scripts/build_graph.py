from pathlib import Path
import argparse
import sys
import pickle

import torch
import networkx as nx

from src.data.graph_builder import load_passes, passes_to_digraph, nx_to_pyg

# Intento opcional de usar la implementación gpickle de NetworkX si existe
try:
    from networkx.readwrite.gpickle import write_gpickle as _nx_write_gpickle
except Exception:  # pragma: no cover
    _nx_write_gpickle = None


def _save_nx_gpickle(G: nx.Graph, path: Path) -> None:
    """Guarda el grafo en formato gpickle de forma robusta."""
    if _nx_write_gpickle is not None:
        _nx_write_gpickle(G, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    ap = argparse.ArgumentParser(description="Construye grafos de pases desde CSV (WhoScored/Eventing2CSV).")
    ap.add_argument("--csv", required=True, help="Ruta al CSV de eventos")
    ap.add_argument("--outdir", default="data/processed", help="Carpeta de salida")
    ap.add_argument("--by-match", default="match_id", help="Columna para separar por partido (o 'none')")
    ap.add_argument("--near-dist", type=float, default=8.0, help="Umbral de cercanía para inferir receptor")
    ap.add_argument("--lookahead", type=int, default=12, help="Filas siguientes a revisar para inferir receptor")
    ap.add_argument("--verbose", action="store_true", help="Imprime detalles del proceso")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] CSV: {csv_path.resolve()}", flush=True)

    # 1) Cargar y filtrar pases (con inferencia de receptor usando el DF completo)
    df = load_passes(csv_path, cm=None, near_dist=args.near_dist, lookahead_rows=args.lookahead)

    total = len(df)
    with_to = int(df["to_id"].notna().sum())
    print(f"[INFO] Pases detectados: {total} | con receptor inferido: {with_to} ({with_to / max(1, total):.1%})", flush=True)

    if args.verbose:
        cm = df.attrs.get("column_map")
        print(
            f"[INFO] Column map: event_type={cm.event_type}, from={cm.player_from}, to={cm.player_to}, "
            f"minute={cm.minute}, team={cm.team_id}, match={cm.match_id}, "
            f"x/y={cm.x_start}/{cm.y_start}, endX/endY={cm.x_end}/{cm.y_end}, "
            f"outcome={cm.outcome}, relatedEvent={cm.related_event}, eventId={cm.event_id}",
            flush=True,
        )

    # Si no hay receptores, sugiere aumentar umbrales y salir
    if with_to == 0:
        print("[WARN] Ningún pase tiene receptor detectable. Prueba aumentar '--near-dist' (10–12) y '--lookahead' (15).", flush=True)
        sys.exit(0)

    # 2) Construir grafos
    by_match = None if args.by_match.lower() == "none" else args.by_match
    graphs = passes_to_digraph(df, by_match=by_match)

    if not graphs:
        print("[WARN] No se generó ningún grafo (diccionario vacío).", flush=True)
        sys.exit(0)

    # 3) Guardar y reportar
    tot_edges = 0
    for mid, G in graphs.items():
        nx_path = outdir / f"passes_{mid}.gpickle"
        pyg_path = outdir / f"passes_{mid}.pt"
        _save_nx_gpickle(G, nx_path)
        data = nx_to_pyg(G)
        torch.save(data, pyg_path)
        tot_edges += G.number_of_edges()
        print(f"[OK] {mid}: nodos={G.number_of_nodes()} aristas={G.number_of_edges()} -> {nx_path.name}, {pyg_path.name}", flush=True)

    print(f"[INFO] Grafos generados: {len(graphs)} | Aristas totales: {tot_edges}", flush=True)


if __name__ == "__main__":
    main()