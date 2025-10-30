from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import math
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data


# ---------- Utilidades ----------

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Devuelve el nombre real (case-insensitive) de la primera columna candidata que exista."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _is_success(val) -> Optional[bool]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "success", "successful", "ok", "completed", "acierto", "exitoso"):
        return True
    if s in ("false", "0", "fail", "failed", "unsuccessful", "miss", "fallo", "no exitoso"):
        return False
    return None


def _euclid(x1, y1, x2, y2) -> float:
    return math.hypot(float(x2) - float(x1), float(y2) - float(y1))


def _pass_distance_angle(
    r: pd.Series,
    xs: Optional[str],
    ys: Optional[str],
    xe: Optional[str],
    ye: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Distancia/ángulo del pase:
      - Si hay start y end -> calcula (dist, ang).
      - Si falta end pero hay start -> (0.0, None) para no introducir NaN.
      - Si falta start o ambos -> (None, None).
    """
    # Validar nombres de columnas
    if not xs or not ys:
        return None, None

    # Tomar valores si la columna existe en la fila
    def _get(col: Optional[str]):
        if col and col in r:
            try:
                v = float(r[col])
                if pd.isna(v):
                    return None
                return v
            except Exception:
                return None
        return None

    xs_v, ys_v = _get(xs), _get(ys)
    xe_v, ye_v = _get(xe), _get(ye)

    # Sin punto de inicio no podemos calcular nada
    if xs_v is None or ys_v is None:
        return None, None

    # Si no hay fin, devolvemos distancia 0 y ángulo indefinido
    if xe_v is None or ye_v is None:
        return 0.0, None

    # Cálculo normal
    dx = xe_v - xs_v
    dy = ye_v - ys_v
    dist = math.hypot(dx, dy)
    ang = math.degrees(math.atan2(dy, dx))  # 0°→derecha, 90°→arriba
    return dist, ang



# ---------- ColumnMap ----------

@dataclass
class ColumnMap:
    event_type: str | None = None
    player_from: str | None = None
    player_to: str | None = None
    minute: str | None = None
    second: str | None = None
    team_id: str | None = None
    match_id: str | None = None
    x_start: str | None = None
    y_start: str | None = None
    x_end: str | None = None
    y_end: str | None = None
    outcome: str | None = None
    related_event: str | None = None
    event_id: str | None = None


def _infer_column_map(df: pd.DataFrame) -> ColumnMap:
    cm = ColumnMap()
    cm.event_type    = _pick(df, ["event_type", "type", "eventType", "event"])
    cm.player_from   = _pick(df, ["player_from", "playerId", "originPlayerId", "playerId_origin", "origin_player_id"])
    cm.player_to     = _pick(df, [
        "player_to", "receiverId",
        "relatedPlayerId",
        "destinationPlayerId", "playerId_target", "target_player_id"
    ])
    cm.minute        = _pick(df, ["minute", "min", "minutes", "eventMinute", "expandedMinute"])
    cm.second        = _pick(df, ["second", "sec", "eventSecond"])
    cm.team_id       = _pick(df, ["team_id", "teamId", "teamID", "team"])
    cm.match_id      = _pick(df, ["match_id", "matchId", "gameId", "fixtureId"])
    cm.x_start       = _pick(df, ["x_start", "x", "startX"])
    cm.y_start       = _pick(df, ["y_start", "y", "startY"])
    cm.x_end         = _pick(df, ["x_end", "endX", "toX", "targetX"])
    cm.y_end         = _pick(df, ["y_end", "endY", "toY", "targetY"])
    cm.outcome       = _pick(df, ["outcome", "outcomeType", "isSuccessful", "success"])
    cm.related_event = _pick(df, ["relatedEventId", "value_RelatedEventId", "type_value_RelatedEventId"])
    cm.event_id      = _pick(df, ["eventId", "id", "event_id"])
    return cm


# ---------- Inferencia de receptores SOBRE EL DF DE EVENTOS COMPLETO ----------

def _infer_receivers_on_events(
    events: pd.DataFrame,
    cm: ColumnMap,
    near_dist: float = 7.5,
    lookahead_rows: int = 10
) -> pd.Series:
    """
    Devuelve una Serie indexada por eventId con el receptor (playerId) del pase.
    Prioridad:
      1) player_to (si existe)
      2) relatedEventId -> playerId del evento relacionado
      3) siguiente evento del MISMO equipo cerca de (endX,endY) y de tipo BallTouch/Carry
    """
    import pandas as pd

    if cm.event_id is None or cm.event_id not in events.columns:
        return pd.Series(dtype="float64")

    # --- ordenar temporalmente para el fallback ---
    ev = events.reset_index(drop=False).rename(columns={"index": "__orig_idx__"})

    order_cols: List[str] = []

    # básicos primero
    if cm.minute and cm.minute in ev.columns:
        order_cols.append(cm.minute)
    if cm.second and cm.second in ev.columns:
        order_cols.append(cm.second)

    # agrega identificadores/tiempo si existen EN ev (no en events)
    for extra in [cm.event_id, "expandedMinute", "eventIndex", "timestamp", "period", "half", "__orig_idx__"]:
        if extra and extra in ev.columns:
            order_cols.append(extra)

    # sanitiza: deja solo columnas válidas y sin duplicados
    seen = set()
    order_cols = [c for c in order_cols if (c not in seen and not seen.add(c))]

    if order_cols:
        ev = ev.sort_values(order_cols).reset_index(drop=True)

    eid_to_rowpos = dict(zip(ev[cm.event_id], ev.index))
    eid_to_pid = dict(zip(events[cm.event_id], events[cm.player_from])) if cm.player_from in events else {}

    # --- 1) receptor directo (deduplicado por eventId) ---
    if cm.player_to and cm.player_to in events.columns:
        tmp = events[[cm.event_id, cm.player_to]].copy()
        tmp = tmp.dropna(subset=[cm.event_id]) \
                 .drop_duplicates(subset=[cm.event_id], keep="last")
        to_direct = pd.Series(tmp[cm.player_to].values, index=tmp[cm.event_id].values, dtype="float64")
    else:
        to_direct = pd.Series(dtype="float64")

    # --- 2) por relatedEventId (deduplicado) ---
        # --- 2) por relatedEventId (deduplicado) + MISMO equipo ---
    if cm.related_event and cm.related_event in events.columns and eid_to_pid:
        # mapas auxiliares
        eid_to_team = (dict(zip(events[cm.event_id], events[cm.team_id]))
                       if cm.team_id and cm.team_id in events.columns else {})

        tmp = events[[cm.event_id, cm.related_event] + ([cm.team_id] if cm.team_id in events.columns else [])].copy()
        tmp = tmp.dropna(subset=[cm.event_id]).drop_duplicates(subset=[cm.event_id], keep="last")

        mapped_pid  = tmp[cm.related_event].map(eid_to_pid)
        if eid_to_team:
            mapped_team = tmp[cm.related_event].map(eid_to_team)          # equipo del evento relacionado
            same_team   = mapped_team.eq(tmp[cm.team_id])                  # ¿coincide con el del pase?
            mapped_pid  = mapped_pid.where(same_team)                      # invalida si es rival

        to_by_related = pd.Series(mapped_pid.values, index=tmp[cm.event_id].values, dtype="float64")
    else:
        to_by_related = pd.Series(dtype="float64")

    # --- 3) fallback espaciotemporal ---
    to_by_near = pd.Series(dtype="float64")
    needed_cols = [cm.team_id, cm.x_end, cm.y_end, cm.event_type, cm.player_from]
    if all(c and c in events.columns for c in needed_cols):
        types_ok = {
        "balltouch","carry","pass",
        "ballreceipt","ball_recovery","touch",
        "reception","control","dribble"
        }
        near_map = {}

        for _, r in events.iterrows():
            if str(r[cm.event_type]).strip().lower() not in ("pass", "pase"):
                continue
            eid = r[cm.event_id]
            if pd.isna(eid):
                continue
            try:
                pos = eid_to_rowpos[eid]
            except KeyError:
                continue

            team = r.get(cm.team_id)
            ex, ey = r.get(cm.x_end), r.get(cm.y_end)
            if pd.isna(team) or pd.isna(ex) or pd.isna(ey):
                continue

            window = ev.iloc[pos + 1 : pos + 1 + lookahead_rows]

            cand = None
            first_same_team_pid = None  # fallback débil

            for _, rr in window.iterrows():
                # 1) Solo consideramos eventos de tu mismo equipo; los del rival se ignoran, NO cortan la búsqueda
                if cm.team_id in rr and pd.notna(rr[cm.team_id]) and rr[cm.team_id] != team:
                    continue  # antes: break

                # 2) Tipo de evento aceptable como “toque/recepción”
                t = str(rr.get(cm.event_type, "")).strip().lower()
                if t not in {"balltouch", "carry", "pass", "ballreceipt", "touch", "ball_recovery"}:
                    continue

                # 3) Coordenadas del “toque” posterior (si no hay startX/startY en tu dataset, usa 'x'/'y')
                sx = cm.x_start or "x"
                sy = cm.y_start or "y"
                rx, ry = rr.get(sx), rr.get(sy)
                if pd.isna(rx) or pd.isna(ry):
                    continue

                # 4) Si este es el PRIMER evento del mismo equipo en la ventana, guárdalo como fallback
                if first_same_team_pid is None and cm.player_from in rr and pd.notna(rr[cm.player_from]):
                    first_same_team_pid = rr.get(cm.player_from)

                # 5) Criterio de cercanía: ¿está el toque cerca del punto de destino del pase?
                if _euclid(ex, ey, rx, ry) <= near_dist and cm.player_from in rr and pd.notna(rr[cm.player_from]):
                    cand = rr.get(cm.player_from)
                    break

            # Si no hubo “cand” por cercanía, usa el primer evento del mismo equipo como fallback débil
            if cand is None and first_same_team_pid is not None and not pd.isna(first_same_team_pid):
                cand = first_same_team_pid

            if cand is not None and not pd.isna(cand):
                near_map[eid] = cand


        if near_map:
            to_by_near = pd.Series(near_map, dtype="float64")

    # --- Combinar con prioridad: directo > related > near ---
    eids_unique = pd.Index(events[cm.event_id].dropna().unique())
    to_final = pd.Series(index=eids_unique, dtype="float64")

    def _merge(src: pd.Series):
        nonlocal to_final
        if src is None or src.empty:
            return
        # deduplicar índice por si acaso y alinear dtype
        src = src[~src.index.duplicated(keep="last")]
        src.index = src.index.astype(to_final.index.dtype, copy=False)
        src = src.reindex(to_final.index)
        mask = to_final.isna() & src.notna()
        to_final.loc[mask] = src.loc[mask]

    _merge(to_direct)
    _merge(to_by_related)
    _merge(to_by_near)

    return to_final


# ---------- Cargador principal ----------

def load_passes(
    csv_path: str | Path,
    cm: ColumnMap | None = None,
    near_dist: float = 7.5,
    lookahead_rows: int = 10
) -> pd.DataFrame:
    """
    Lee el CSV completo de eventos, filtra pases y añade:
      - from_id (playerId del pasador)
      - to_id   (receptor, inferido si falta)
      - pass_dist, pass_ang, pass_success
    """
    # intenta UTF-8, si falla latin-1
    try:
        events = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        events = pd.read_csv(csv_path, encoding="latin-1", on_bad_lines="skip")

    if cm is None:
        cm = _infer_column_map(events)

    # columnas clave mínimas
    core_needed = ["event_type", "player_from"]
    missing = [k for k in core_needed if getattr(cm, k) is None]
    if missing:
        raise ValueError(f"No se pudieron detectar columnas clave {missing}. Cols: {list(events.columns)}")

    # subset de PASES
    ev_col = cm.event_type
    mask_pass = events[ev_col].astype(str).str.lower().isin(["pass", "pase"])
    df = events[mask_pass].copy()

    # normaliza tipos
    for col in [cm.player_from, cm.team_id]:
        if col and col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if cm.minute and cm.minute in df:
        df[cm.minute] = pd.to_numeric(df[cm.minute], errors="coerce").fillna(0).astype(int)

    # outcome → bool
    if cm.outcome and cm.outcome in df:
        df["pass_success"] = df[cm.outcome].map(_is_success)
    else:
        df["pass_success"] = None

    # distancia/ángulo (para cada pase)
    dist_ang = df.apply(
        lambda r: _pass_distance_angle(r, cm.x_start, cm.y_start, cm.x_end, cm.y_end),
        axis=1,
        result_type="expand",
    )
    if isinstance(dist_ang, pd.DataFrame):
        df["pass_dist"] = dist_ang[0]
        df["pass_ang"] = dist_ang[1]
    else:
        df["pass_dist"] = None
        df["pass_ang"] = None

    # from_id
    df["from_id"] = pd.to_numeric(df[cm.player_from], errors="coerce")

    # to_id inferido en el DF COMPLETO y luego mapeado a los pases por eventId
    if cm.event_id and cm.event_id in events.columns:
        to_series = _infer_receivers_on_events(
            events, cm, near_dist=near_dist, lookahead_rows=lookahead_rows
        )  # index: eventId
        if cm.event_id in df.columns:
            df["to_id"] = df[cm.event_id].map(to_series)
        else:
            # si por algún motivo df perdió el eventId, intentamos por índice
            df = df.reset_index(drop=False).rename(columns={"index": "__orig_idx__"})
            df["to_id"] = pd.NA
    else:
        # sin eventId: peor caso, no inferimos (se puede extender)
        df["to_id"] = pd.NA

    # guarda el mapeo detectado
    df.attrs["column_map"] = cm
    return df


# ---------- Construcción de grafos ----------

def passes_to_digraph(
    df: pd.DataFrame,
    cm: ColumnMap | None = None,
    by_match: Optional[str] = None,
) -> Dict[str, nx.DiGraph]:
    """Retorna {match_id: DiGraph}. Si no hay match_id, usa 'GLOBAL'."""
    if cm is None:
        cm = df.attrs.get("column_map") or _infer_column_map(df)

    graphs: Dict[str, nx.DiGraph] = {}

    # agrupar por partido (si procede)
    if by_match and by_match in df.columns:
        groups: Iterable[Tuple[str, pd.DataFrame]] = df.groupby(by_match)
    elif cm.match_id and cm.match_id in df.columns:
        groups = df.groupby(cm.match_id)
    else:
        groups = [("GLOBAL", df)]

    for mid, part in groups:
        G = nx.DiGraph(match_id=str(mid))

        # ---- NODOS ----
        players = pd.concat([part["from_id"], part["to_id"]]).dropna().unique().tolist()
        for pid in players:
            G.add_node(int(pid))

        # ---- VOTOS DE EQUIPO ----
        # 1) equipo del EMISOR (mayoría por from_id)
        sender_team = {}
        if cm.team_id and cm.team_id in part.columns:
            tmp = part[["from_id", cm.team_id]].dropna()
            if not tmp.empty:
                tmp["from_id"] = pd.to_numeric(tmp["from_id"], errors="coerce").astype("Int64")
                s_map = (tmp.dropna()
                            .groupby("from_id")[cm.team_id]
                            .agg(lambda s: s.value_counts().idxmax()))
                sender_team = {int(k): int(v) for k, v in s_map.to_dict().items()
                               if pd.notna(k) and pd.notna(v)}

        # 2) votos para el RECEPTOR (hereda el equipo del emisor en cada pase)
        recv_votes: Dict[int, List[int]] = {}

        # ---- ARISTAS ----
        for _, r in part.iterrows():
            a, b = r.get("from_id"), r.get("to_id")
            if pd.isna(a) or pd.isna(b):
                continue
            a, b = int(a), int(b)

            # voto de equipo para b según el equipo del pase
            if cm.team_id and cm.team_id in part.columns and pd.notna(r.get(cm.team_id)):
                t = int(r.get(cm.team_id))
                recv_votes.setdefault(b, []).append(t)

            w_prev = G[a][b]["weight"] if G.has_edge(a, b) else 0
            succ_prev = G[a][b].get("success_count", 0) if G.has_edge(a, b) else 0
            dist_prev = G[a][b].get("dist_sum", 0.0) if G.has_edge(a, b) else 0.0
            mins_prev = G[a][b].get("minutes", []) if G.has_edge(a, b) else []

            w = w_prev + 1
            succ = succ_prev + (1 if r.get("pass_success") else 0)
            dist_sum = dist_prev + (float(r["pass_dist"]) if pd.notna(r["pass_dist"]) else 0.0)
            minute_v = int(r.get("minute", 0)) if "minute" in r else 0
            minutes = mins_prev + [minute_v]

            G.add_edge(
                a, b,
                weight=w,
                success_count=succ,
                dist_sum=dist_sum,
                minutes=minutes,
            )

        # ---- ATRIBUTO DE NODO: team ----
        team_map = dict(sender_team)
        for pid, votes in recv_votes.items():
            if votes:
                team_map[pid] = max(set(votes), key=votes.count)  # modo

        if team_map:
            nx.set_node_attributes(G, name="team", values=team_map)

        # --- Filtrar aristas cruzadas (entre equipos distintos o sin team) ---
        removed = 0
        for u, v in list(G.edges()):
            tu = team_map.get(u, None)
            tv = team_map.get(v, None)
            if tu is None or tv is None or tu != tv:
                G.remove_edge(u, v)
                removed += 1
        
        G.graph["removed_cross_team"] = removed
        G.graph["edges_after_filter"] = G.number_of_edges()
        
        # +++ MÉTRICAS DE COBERTURA DE TEAM EN EL GRAFO +++
        has_team = [n for n, d in G.nodes(data=True) if "team" in d]
        G.graph["team_coverage_pct"] = (100.0 * len(has_team) / max(1, G.number_of_nodes()))

        graphs[str(mid)] = G

    return graphs

def nx_to_pyg(G: nx.DiGraph) -> Data:
    """Convierte un DiGraph en un Data de PyG con features simples (in/out degree)."""
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # edges
    src, dst, w, dist_avg = [], [], [], []
    for u, v, d in G.edges(data=True):
        src.append(node_to_idx[u])
        dst.append(node_to_idx[v])
        w.append(float(d.get("weight", 1)))
        dist_sum = float(d.get("dist_sum", 0.0))
        dist_avg.append(dist_sum / d["weight"] if d.get("weight", 0) else 0.0)

    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if len(src)
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(list(zip(w, dist_avg)), dtype=torch.float) if len(w) else None
    )

    # features: in/out degree
    in_deg = (
        torch.tensor([G.in_degree(n) for n in nodes], dtype=torch.float).view(-1, 1)
        if nodes else torch.zeros((0, 1))
    )
    out_deg = (
        torch.tensor([G.out_degree(n) for n in nodes], dtype=torch.float).view(-1, 1)
        if nodes else torch.zeros((0, 1))
    )
    x = torch.cat([in_deg, out_deg], dim=1) if nodes else torch.zeros((0, 2))

    data = Data(x=x, edge_index=edge_index)
    if edge_attr is not None:
        data.edge_attr = edge_attr  # [E,2] → weight, dist_avg
    data.node_ids = torch.tensor(nodes, dtype=torch.long) if nodes else torch.tensor([], dtype=torch.long)
    return data