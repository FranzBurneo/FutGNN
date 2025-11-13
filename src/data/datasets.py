from __future__ import annotations

import glob
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset

from src.data.snapshots import build_snapshots_for_match
from src.tasks.labels import build_lp_labels

class SnapshotLPDataset(Dataset):
    """
    Dataset para Link Prediction (predecir receptor) a partir de snapshots temporales.

    __getitem__ -> (pyg, pos_edges, neg_edges, mask, meta)
      - pyg: torch_geometric.data.Data (grafo del snapshot)
      - pos_edges: Tensor[2, P] aristas positivas (u->v reales en t+1)
      - neg_edges: Tensor[2, N] aristas negativas (u->v muestreadas)
      - mask: Tensor(bool) indica si este snapshot tiene positivos válidos
      - meta: dict con metadatos (ej. t_start, t_end, etc.)
    """

    def __init__(
        self,
        files_or_glob: List[str] | str,
        step: int = 1,
        negative_k: int = 10,
    ) -> None:
        super().__init__()

        # Admite lista de archivos o un patrón glob
        if isinstance(files_or_glob, str):
            self.files: List[str] = sorted(glob.glob(files_or_glob))
        else:
            self.files = list(files_or_glob)

        assert self.files, "SnapshotLPDataset: no se encontraron archivos CSV."

        self.step = step
        self.negative_k = negative_k

        # Construimos todos los snapshots una vez
        self.samples: List[Tuple[Any, Dict[str, Any]]] = []
        for f in self.files:
            snaps = build_snapshots_for_match(f, step=self.step)
            # snaps: List[(pyg, meta)]
            for pyg, meta in snaps:
                # Saltar snapshots vacíos (snapshots.py devuelve Tensor vacío en ese caso)
                if isinstance(pyg, torch.Tensor) and pyg.numel() == 0:
                    continue
                self.samples.append((pyg, meta))

        assert len(self.samples) > 0, "No se generaron snapshots. Revisa snapshots.py y los CSV."

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        pyg, meta = self.samples[idx]
        pos, neg, mask = build_lp_labels(pyg, meta=meta, negative_k=self.negative_k)
        # Orden consistente con lo que venimos usando:
        return pyg, pos, neg, mask, meta