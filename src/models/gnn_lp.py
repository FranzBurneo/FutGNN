from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, SAGEConv


def _make_conv(kind: str, in_ch: int, out_ch: int):
    """
    Pequeña factory para elegir el tipo de capa GNN.
    kind: "gcn" | "sage"
    """
    k = (kind or "gcn").lower()
    if k == "gcn":
        return GCNConv(in_ch, out_ch)
    if k == "sage":
        return SAGEConv(in_ch, out_ch)
    raise ValueError(f"Conv desconocida: {kind}. Use 'gcn' o 'sage'.")


class EdgeScorer(nn.Module):
    """
    Puntuador de pares (u, v) a partir de embeddings de nodos.

    mode:
      - "dot":      producto punto (baseline simple)
      - "bilinear": capa bilinear (suele mejorar respecto a dot)
      - "mlp":      concatenación [z_u||z_v] + MLP pequeño
    """
    def __init__(self, mode: str, dim: int, mlp_hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        mode = (mode or "dot").lower()
        assert mode in {"dot", "bilinear", "mlp"}
        self.mode = mode

        if mode == "bilinear":
            self.bilin = nn.Bilinear(dim, dim, 1)
        elif mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(2 * dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
            )

    def forward(self, z: Tensor, edge_pairs: Tensor) -> Tensor:
        """
        z: [N, D]
        edge_pairs: [2, K] con índices (u, v)
        return: logits [K]
        """
        if edge_pairs.ndim == 1:
            edge_pairs = edge_pairs.view(2, 1)

        u = edge_pairs[0].long()
        v = edge_pairs[1].long()

        zu = z[u]  # [K, D]
        zv = z[v]  # [K, D]

        if self.mode == "dot":
            return (zu * zv).sum(dim=-1)                       # [K]
        elif self.mode == "bilinear":
            return self.bilin(zu, zv).squeeze(-1)              # [K]
        else:  # "mlp"
            x = torch.cat([zu, zv], dim=-1)                    # [K, 2D]
            return self.mlp(x).squeeze(-1)                     # [K]


class GNNSimpleLP(nn.Module):
    """
    Modelo GNN para Link Prediction (predicción de receptor) sobre snapshots de redes de pases.

    - encode(data) -> z: embeddings de nodos
    - score_pairs(z, edge_pairs) -> logits para (u->v)
    - score_edges(data, edge_pairs) -> encode + score_pairs (útil en evaluación)
    - loss(data, pos, neg) -> BCEWithLogits sobre positivos/negativos

    Parámetros clave:
      * conv_type: "gcn" (por defecto) o "sage" para GraphSAGE.
      * scorer: "dot" | "bilinear" | "mlp" (bilinear recomendado como baseline fuerte).
    """
    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        layers: int = 2,
        out: int = 64,
        dropout: float = 0.1,
        scorer: str = "bilinear",
        scorer_mlp_hidden: int = 64,
        conv_type: str = "gcn",
    ):
        super().__init__()
        assert layers >= 1, "Debe haber al menos 1 capa GNN."
        self.layers = layers
        self.dropout_p = dropout
        self.conv_type = conv_type

        # Backbone GNN (seleccionable)
        self.convs = nn.ModuleList()
        self.convs.append(_make_conv(conv_type, in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(_make_conv(conv_type, hidden, hidden))

        # Proyección final al espacio 'out'
        self.proj = nn.Linear(hidden, out)

        # Activación/Dropout
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Scorer configurable
        self.scorer = EdgeScorer(
            mode=scorer,
            dim=out,
            mlp_hidden=scorer_mlp_hidden,
            dropout=dropout,
        )

    def encode(self, data) -> Tensor:
        """
        Espera un torch_geometric.data.Data con:
          - x: [N, in_dim]
          - edge_index: [2, E]
        Devuelve:
          - z: [N, out]
        """
        x, edge_index = data.x, data.edge_index
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)
        z = self.proj(h)  # [N, out]
        return z

    def score_pairs(self, z: Tensor, edge_pairs: Tensor) -> Tensor:
        """
        z: [N, out]
        edge_pairs: [2, K]
        return: logits [K]
        """
        return self.scorer(z, edge_pairs)

    @torch.no_grad()
    def score_edges(self, data, edge_pairs: Tensor) -> Tensor:
        """
        encode(data) + score_pairs, útil para evaluación/inferencia.
        Devuelve logits (NO aplica sigmoide).
        """
        z = self.encode(data)
        return self.score_pairs(z, edge_pairs)

    def loss(self, data, pos: Tensor, neg: Tensor) -> Tensor:
        """
        Pérdida binaria con logits para pos/neg.
          - pos: [2, P]
          - neg: [2, N]
        """
        z = self.encode(data)
        pos_logits = self.score_pairs(z, pos)  # [P]
        neg_logits = self.score_pairs(z, neg)  # [N]

        logits = torch.cat([pos_logits, neg_logits], dim=0)  # [P+N]
        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits)
        ], dim=0)

        criterion = nn.BCEWithLogitsLoss()
        return criterion(logits, labels)

    def forward(self, data) -> Tensor:
        """
        No se usa directamente para LP (se emplea loss/score_edges),
        pero se deja por compatibilidad (devuelve z).
        """
        return self.encode(data)
