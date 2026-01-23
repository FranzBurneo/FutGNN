from __future__ import annotations
import torch
from collections import Counter

def _team_of(node_idx: int, teams: torch.Tensor) -> int:
    # teams: [N] con id de equipo por nodo (int); si no existe, retorna 0
    return int(teams[node_idx].item()) if teams is not None else 0

def candidates_same_team(u_idx: int, teams: torch.Tensor, num_nodes: int):
    tu = _team_of(u_idx, teams)
    return [v for v in range(num_nodes) if v != u_idx and _team_of(v, teams) == tu]

def score_indegree(pyg, u_idx: int) -> list[tuple[int, float]]:
    # in_degree a partir de edge_index (snapshot G_t)
    E = pyg.edge_index
    num_nodes = pyg.num_nodes
    indeg = torch.bincount(E[1], minlength=num_nodes).float()
    teams = getattr(pyg, "teams", None)
    cand = candidates_same_team(u_idx, teams, num_nodes)
    return [(v, float(indeg[v].item())) for v in cand]

def score_uv_frequency(pyg, u_idx: int) -> list[tuple[int, float]]:
    # frecuencia histórica u->v en snapshot
    E = pyg.edge_index
    num_nodes = pyg.num_nodes
    teams = getattr(pyg, "teams", None)
    cand = candidates_same_team(u_idx, teams, num_nodes)
    cnt = Counter()
    src = E[0].tolist(); dst = E[1].tolist()
    for uu, vv in zip(src, dst):
        if uu == u_idx:
            cnt[vv] += 1
    return [(v, float(cnt[v])) for v in cand]

def score_common_neighbors(pyg, u_idx: int) -> list[tuple[int, float]]:
    # |Γ(u) ∩ Γ(v)| con Γ saliente en G_t (puedes probar también entrante)
    E = pyg.edge_index
    num_nodes = pyg.num_nodes
    teams = getattr(pyg, "teams", None)
    cand = candidates_same_team(u_idx, teams, num_nodes)

    outs = [[] for _ in range(num_nodes)]
    for uu, vv in zip(E[0].tolist(), E[1].tolist()):
        outs[uu].append(vv)

    Nu = set(outs[u_idx])
    scores = []
    for v in cand:
        Nv = set(outs[v])
        scores.append((v, float(len(Nu.intersection(Nv)))))
    return scores

def score_random_team(pyg, u_idx: int) -> list[tuple[int, float]]:
    # puntajes iguales => ranking aleatorio pero reproducible si fijas la semilla fuera
    num_nodes = pyg.num_nodes
    teams = getattr(pyg, "teams", None)
    cand = candidates_same_team(u_idx, teams, num_nodes)
    return [(v, 0.0) for v in cand]  # todos 0 => desempate aleatorio externo
