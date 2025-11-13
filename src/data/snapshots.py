from __future__ import annotations
from typing import List, Tuple, Dict, Any, Iterable, Optional

import pandas as pd
import networkx as nx
import torch

from src.data.graph_builder import load_passes, passes_to_digraph, nx_to_pyg


def _slice_by_minute(df: pd.DataFrame, minute_col: str, t0: int, t1: int) -> pd.DataFrame:
    m = pd.to_numeric(df[minute_col], errors="coerce").fillna(0).astype(int)
    return df.loc[(m >= t0) & (m < t1)].copy()


def _graph_from_slice(df_slice: pd.DataFrame) -> nx.DiGraph:
    # df_slice ya es un solo “partido/ventana”, así que forzamos agrupación GLOBAL
    G_dict = passes_to_digraph(df_slice, cm=df_slice.attrs.get("column_map"), by_match=None)
    # passes_to_digraph devolverá {"GLOBAL": G} en este caso
    return next(iter(G_dict.values()))


def _edge_index_with_mapping(G_src: nx.DiGraph, G_next: nx.DiGraph) -> torch.Tensor:
    """
    Convierte las aristas de G_next al índice de nodos de G_src,
    usando los player_ids (nodos de NetworkX) como ancla.
    Descartamos aristas cuyos nodos no existan en G_src.
    """
    # índices del snapshot actual
    nodes_src = list(G_src.nodes())
    idx_src = {n: i for i, n in enumerate(nodes_src)}

    src_u, src_v = [], []
    for u, v in G_next.edges():
        if (u in idx_src) and (v in idx_src):
            src_u.append(idx_src[u])
            src_v.append(idx_src[v])

    if not src_u:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor([src_u, src_v], dtype=torch.long)


def build_snapshots_for_match(
    match_csv_path: str,
    step: int = 5,
    near_dist: float = 7.5,
    lookahead: int = 10
) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Devuelve lista de tuples (pyg_data, meta) por ventanas de tiempo:
      - pyg_data: Data (PyG) con x, edge_index, edge_attr, node_ids
      - meta: { 't_start', 't_end', 'future_edge_index' (si hay siguiente ventana) }

    El 'future_edge_index' está expresado en el índice de nodos del snapshot actual.
    """
    # 1) Leer y preparar pases
    df = load_passes(match_csv_path, cm=None, near_dist=near_dist, lookahead_rows=lookahead)
    cm = df.attrs.get("column_map")
    minute_col = (cm.minute if cm and cm.minute in df.columns else None) or (
        "minute" if "minute" in df.columns else None
    )
    if minute_col is None:
        # Si no existe, asumimos todo en minuto 0 (una sola ventana)
        df["__minute__"] = 0
        minute_col = "__minute__"

    # normalizar a int
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce").fillna(0).astype(int)
    max_minute = int(df[minute_col].max()) if not df.empty else 0

    # 2) Construir ventanas
    windows: List[Tuple[int, int]] = []
    t = 0
    if step <= 0:
        step = 5
    while t <= max_minute:
        windows.append((t, min(t + step, max_minute + 1)))
        t += step

    # 3) Snapshot por ventana
    snaps: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    graphs: List[Optional[nx.DiGraph]] = []

    for (t0, t1) in windows:
        part = _slice_by_minute(df, minute_col, t0, t1)
        if part.empty:
            graphs.append(None)
            snaps.append((torch.empty(0), {"t_start": t0, "t_end": t1, "future_edge_index": None}))
            continue

        G = _graph_from_slice(part)
        pyg = nx_to_pyg(G)
        meta: Dict[str, Any] = {"t_start": t0, "t_end": t1, "future_edge_index": None}
        snaps.append((pyg, meta))
        graphs.append(G)

    # 4) Construir future_edge_index usando la siguiente ventana
    for i in range(len(snaps) - 1):
        pyg_i, meta_i = snaps[i]
        G_i = graphs[i]
        G_j = graphs[i + 1]
        if (G_i is None) or (G_j is None):
            continue
        # mapear aristas del siguiente bloque al índice de nodos del actual
        fei = _edge_index_with_mapping(G_i, G_j)
        if fei.numel() > 0:
            meta_i["future_edge_index"] = fei
        else:
            meta_i["future_edge_index"] = None

    return snaps