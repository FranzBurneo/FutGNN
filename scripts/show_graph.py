# scripts/show_graph.py
from __future__ import annotations

import argparse
import os
import pickle
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# ------------ Carga compatible de gpickle (NX 2.x / 3.x) ------------
def _read_gpickle_compat(path: str):
    try:
        if hasattr(nx, "read_gpickle"):
            return nx.read_gpickle(path)  # NetworkX 2.x
        else:
            from networkx.readwrite.gpickle import read_gpickle  # type: ignore
            return read_gpickle(path)  # NetworkX 3.x
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------ Utilidades de selección ------------
def _load_graph_from_report(report_csv: str, match: Optional[str], index: Optional[int]) -> Tuple[nx.DiGraph, str]:
    df = pd.read_csv(report_csv)

    if match:
        # Búsqueda no-regex para evitar sorpresas
        rowset = df[df["csv"].str.contains(match, case=False, regex=False)]
        if rowset.empty:
            raise ValueError(f"No se encontró un partido que contenga: {match}")
        row = rowset.iloc[0]
    elif index is not None:
        if index < 0 or index >= len(df):
            raise IndexError(f"--index fuera de rango (0..{len(df)-1})")
        row = df.iloc[index]
    else:
        raise ValueError("Debes pasar --match o --index")

    gpath = row["nx_path"]
    if not os.path.exists(gpath):
        raise FileNotFoundError(f"No existe el archivo de grafo: {gpath}")

    G = _read_gpickle_compat(gpath)
    title = str(row["csv"])
    return G, title


def _filter_by_weight(G: nx.DiGraph, min_weight: int) -> nx.DiGraph:
    if min_weight <= 1:
        return G
    H = G.copy()
    to_remove = [(u, v) for u, v, d in H.edges(data=True) if int(d.get("weight", 1)) < min_weight]
    H.remove_edges_from(to_remove)
    # Remueve nodos aislados resultantes
    isolates = list(nx.isolates(H))
    H.remove_nodes_from(isolates)
    return H


def _limit_top_edges(G: nx.DiGraph, top_n: int) -> nx.DiGraph:
    if top_n is None or top_n <= 0:
        return G
    # Ordena por peso descendente
    edges_sorted = sorted(G.edges(data=True), key=lambda e: int(e[2].get("weight", 1)), reverse=True)
    keep = edges_sorted[:top_n]
    H = nx.DiGraph()
    for u, v, d in keep:
        H.add_edge(u, v, **d)
    # Asegura nodos
    for u, v, _ in keep:
        if u not in H:
            H.add_node(u, **G.nodes[u])
        if v not in H:
            H.add_node(v, **G.nodes[v])
    return H


# ------------ Dibujo ------------
def _compute_layout(G: nx.DiGraph, layout: str, seed: int = 42):
    if layout == "spring":
        return nx.spring_layout(G, seed=seed, k=None)
    if layout == "kamada":
        return nx.kamada_kawai_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "spectral":
        return nx.spectral_layout(G)
    # por defecto
    return nx.spring_layout(G, seed=seed)


def _draw_graph(
    G: nx.DiGraph,
    title: str,
    layout: str = "spring",
    show_ids: bool = False,
    annotate_edges: bool = False,
    arrow_scale: float = 12.0,
    seed: int = 42,
):
    if G.number_of_nodes() == 0:
        raise ValueError("El grafo no tiene nodos después de los filtros.")

    pos = _compute_layout(G, layout=layout, seed=seed)

    # Colores por equipo (atributo de nodo 'team'); nodos sin 'team' → gris
    teams = nx.get_node_attributes(G, "team")
    unique_teams = sorted(set(teams.values())) if teams else []
    palette = {}
    # Usamos colores por defecto de Matplotlib ciclando
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, t in enumerate(unique_teams):
        palette[t] = default_colors[i % max(1, len(default_colors))]
    default_node_color = "#9e9e9e"

    node_colors = []
    for n in G.nodes():
        t = teams.get(n, None)
        node_colors.append(palette.get(t, default_node_color))

    # Grosor de arista por peso; alpha para legibilidad
    weights = [int(d.get("weight", 1)) for _, _, d in G.edges(data=True)]
    if len(weights):
        max_w = max(weights)
        widths = [1.0 + 3.0 * (w / max(1, max_w)) for w in weights]
        alphas = [0.25 + 0.75 * (w / max(1, max_w)) for w in weights]
    else:
        widths = []
        alphas = []

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(
        G, pos,
        node_size=650,
        node_color=node_colors,
        edgecolors="#333333",
        linewidths=1.0,
    )
    # Dibuja aristas con alfa proporcional
    for (u, v, d), w, a in zip(G.edges(data=True), widths, alphas):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=w,
            alpha=a,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=arrow_scale,
            connectionstyle="arc3,rad=0.08",  # leve curvatura
        )

    # Etiquetas de nodos
    if show_ids:
        labels = {n: str(n) for n in G.nodes()}
    else:
        # Si tienes un mapeo de nombres, podrías integrarlo aquí.
        labels = {n: str(n) for n in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold")

    # Etiquetas de aristas (peso)
    if annotate_edges:
        edge_labels = {(u, v): int(d.get("weight", 1)) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=False)

    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()


# ------------ CLI ------------
def parse_args():
    p = argparse.ArgumentParser(description="Mostrar grafo de pases desde report_enhanced.csv")
    p.add_argument("--report", default="data/processed/report_enhanced.csv", help="Ruta al CSV de reporte")
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--match", type=str, help="Cadena que debe contener el nombre del partido (columna csv)")
    sel.add_argument("--index", type=int, help="Índice de fila en el CSV (0-based)")
    p.add_argument("--min-weight", type=int, default=1, help="Filtra aristas con peso < min-weight")
    p.add_argument("--top-n", type=int, default=0, help="Muestra solo las top-N aristas por peso (0 = sin límite)")
    p.add_argument("--layout", choices=["spring", "kamada", "circular", "spectral"], default="spring", help="Layout del grafo")
    p.add_argument("--show-ids", action="store_true", help="Mostrar IDs numéricos de los nodos")
    p.add_argument("--annotate-edges", action="store_true", help="Poner etiquetas con el peso en las aristas")
    p.add_argument("--seed", type=int, default=42, help="Semilla para layouts aleatorios")
    p.add_argument("--save", type=str, default="", help="Ruta para guardar PNG en vez de mostrar en pantalla")
    return p.parse_args()


def main():
    args = parse_args()
    G, title = _load_graph_from_report(args.report, args.match, args.index)

    # Filtros opcionales
    if args.min_weight and args.min_weight > 1:
        G = _filter_by_weight(G, args.min_weight)
    if args.top_n and args.top_n > 0:
        G = _limit_top_edges(G, args.top_n)

    _draw_graph(
        G,
        title=title,
        layout=args.layout,
        show_ids=args.show_ids,
        annotate_edges=args.annotate_edges,
        arrow_scale=12.0,
        seed=args.seed,
    )

    if args.save:
        out = args.save
        os.makedirs(os.path.dirname(out), exist_ok=True) if os.path.dirname(out) else None
        plt.savefig(out, dpi=200)
        print(f"[OK] Imagen guardada en: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
