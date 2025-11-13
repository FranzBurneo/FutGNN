from __future__ import annotations
import torch

def _iter_samples(dl):
    """
    Normaliza la iteración sobre el DataLoader:
    - Si el collate devuelve [pyg_list, pos_list, neg_list, mask_list, meta_list],
      renderea (pyg, pos, neg, mask, meta) uno a uno.
    - Si algún día cambias el collate a devolver ya tuplas por sample,
      también lo soporta.
    """
    for batch in dl:
        # caso collate = list(zip(*batch))
        if isinstance(batch, (list, tuple)) and len(batch) == 5 and isinstance(batch[0], (list, tuple)):
            pyg_list, pos_list, neg_list, mask_list, meta_list = batch
            for pyg, pos, neg, mask, meta in zip(pyg_list, pos_list, neg_list, mask_list, meta_list):
                yield pyg, pos, neg, mask, meta
        # caso "batch" ya es una tupla/single sample
        elif isinstance(batch, (list, tuple)) and len(batch) == 5:
            yield batch
        else:
            raise RuntimeError(f"Formato de batch no soportado: {type(batch)} con len={len(batch) if hasattr(batch,'__len__') else 'NA'}")

def train_one_epoch(model, dl, optim, device: str):
    model.train()
    total = 0.0
    n = 0
    for pyg, pos, neg, mask, _meta in _iter_samples(dl):
        pyg  = pyg.to(device)
        pos  = pos.to(device)
        neg  = neg.to(device)
        # mask puede venir como escalar bool; si existe y es False, salta
        if mask is not None and hasattr(mask, "item") and not bool(mask.item()):
            continue

        loss = model.loss(pyg, pos, neg)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total += float(loss.item())
        n += 1
    return total / max(1, n)

@torch.no_grad()
def eval_hits_at_k(model, dl, K=5, device: str = "cpu"):
    model.eval()
    hits = 0
    total = 0
    for pyg, pos, neg, mask, _meta in _iter_samples(dl):
        pyg  = pyg.to(device)
        pos  = pos.to(device)
        neg  = neg.to(device)
        if mask is not None and hasattr(mask, "item") and not bool(mask.item()):
            continue

        # score del positivo vs negativos
        # (asumimos pos es una única arista [2] o [2,1]; adaptamos forma)
        if pos.ndim == 1:
            pos = pos.view(2, 1)
        # concat candidatos: primero el positivo, luego los negativos
        candidates = torch.cat([pos, neg], dim=1)  # [2, 1+N]

        scores = model.score_edges(pyg, candidates)  # [1+N]
        # rank del positivo (índice 0)
        rank = (scores[0] <= scores).sum().item()  # 1 = mejor
        if rank <= K:
            hits += 1
        total += 1

    return hits / max(1, total)