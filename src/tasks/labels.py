from __future__ import annotations

from typing import Tuple, Dict, Any
import torch

try:
    # Usamos la util de PyG si está disponible (más eficiente)
    from torch_geometric.utils import negative_sampling
    _HAS_NEG = True
except Exception:
    _HAS_NEG = False


def _dedup_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Quita duplicados de edge_index (2, E) manteniendo el orden relativo.
    """
    if edge_index.numel() == 0:
        return edge_index
    # Convertimos a tuplas para deduplicar columnas
    ei = edge_index.t().contiguous()
    # unique con return_inverse preservando primer índice
    # (Si tu versión de torch no tiene return_inverse en unique, caemos a set)
    try:
        uniq, idx = torch.unique(ei, dim=0, return_inverse=True)
        # 'uniq' ya está deduplicado en orden de aparición
        return uniq.t().contiguous()
    except Exception:
        seen = set()
        rows = []
        for u, v in ei.tolist():
            key = (int(u), int(v))
            if key not in seen:
                rows.append([u, v])
                seen.add(key)
        return torch.tensor(rows, dtype=edge_index.dtype, device=edge_index.device).t().contiguous()


def _neg_sample_fallback(pos: torch.Tensor, num_nodes: int, num_neg: int) -> torch.Tensor:
    """
    Negativos por muestreo uniforme evitando self-loops y aristas positivas.
    pos: (2, P)
    return: (2, num_neg)
    """
    device = pos.device
    pos_set = set((int(u), int(v)) for u, v in pos.t().tolist())
    result = []
    tries = 0
    max_tries = max(1000, num_neg * 10)

    import random
    while len(result) < num_neg and tries < max_tries:
        u = random.randrange(0, num_nodes)
        v = random.randrange(0, num_nodes)
        tries += 1
        if u == v:
            continue
        if (u, v) in pos_set:
            continue
        result.append([u, v])

    if not result:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    neg = torch.tensor(result[:num_neg], dtype=torch.long, device=device).t().contiguous()
    return neg


def build_lp_labels(
    pyg,
    meta: Dict[str, Any] | None = None,
    negative_k: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Genera etiquetas para Link Prediction.

    Devuelve:
      pos: Tensor[2, P]  (aristas positivas)
      neg: Tensor[2, N]  (aristas negativas, N = negative_k * P)
      mask: Tensor(bool)  True si hay al menos 1 arista positiva

    Prioridad para positivas:
      1) meta["future_edge_index"] si existe (snapshot t vs. t+1)
      2) pyg.target_edge_index si existe
      3) pyg.edge_index (aristas del snapshot actual)
    """
    if meta is None:
        meta = {}

    # 1) Positivas (en orden de prioridad)
    pos = None
    if "future_edge_index" in meta and meta["future_edge_index"] is not None:
        pos = meta["future_edge_index"]
    elif hasattr(pyg, "target_edge_index") and getattr(pyg, "target_edge_index") is not None:
        pos = getattr(pyg, "target_edge_index")
    else:
        pos = getattr(pyg, "edge_index", None)

    if pos is None:
        # Caso extremo: no hay manera de obtener positivas
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, empty, torch.tensor(False)

    pos = pos.long().contiguous()
    if pos.ndim != 2 or pos.size(0) != 2:
        raise ValueError("build_lp_labels: edge_index esperado con shape [2, E]")

    # Deduplicar y filtrar self-loops
    pos = _dedup_edges(pos)
    if pos.numel() > 0:
        mask_noloop = pos[0] != pos[1]
        pos = pos[:, mask_noloop]

    P = pos.size(1)
    if P == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=pos.device)
        return empty, empty, torch.tensor(False, device=pos.device)

    # 2) Negativas
    num_nodes = pyg.num_nodes if hasattr(pyg, "num_nodes") else int(pyg.x.size(0))
    num_neg = int(max(1, negative_k) * P)

    if _HAS_NEG:
        try:
            neg = negative_sampling(
                edge_index=pos,      # muestreamos vs. positivas reales
                num_nodes=num_nodes,
                num_neg_samples=num_neg,
                method='sparse'      # si tu versión no soporta, PyG caerá a 'dense' o lanza exception
            )
        except Exception:
            neg = _neg_sample_fallback(pos, num_nodes, num_neg)
    else:
        neg = _neg_sample_fallback(pos, num_nodes, num_neg)

    # Asegurar que no haya self-loops en negativos
    if neg.numel() > 0:
        mask_noloop = neg[0] != neg[1]
        neg = neg[:, mask_noloop]

    # 3) Mask
    mask = torch.tensor(True, device=pos.device)

    return pos, neg, mask


__all__ = ["build_lp_labels"]