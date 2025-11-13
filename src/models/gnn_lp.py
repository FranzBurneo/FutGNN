from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class EdgeScorer(nn.Module):
    """
    Módulo para puntuar pares (u,v) a partir de embeddings de nodos.
    modes:
      - "dot":       producto punto (baseline)
      - "bilinear":  capa bilinear (rápida y suele mejorar bastante)
      - "mlp":       concatenación + MLP pequeño (más expresivo)
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

    def forward(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        if edge_pairs.ndim == 1:
            edge_pairs = edge_pairs.view(2, 1)
        u = edge_pairs[0].long()
        v = edge_pairs[1].long()

        zu = z[u]  # [K, D]
        zv = z[v]  # [K, D]

        if self.mode == "dot":
            return (zu * zv).sum(dim=-1)  # [K] logits
        elif self.mode == "bilinear":
            return self.bilin(zu, zv).squeeze(-1)  # [K]
        else:  # "mlp"
            x = torch.cat([zu, zv], dim=-1)       # [K, 2D]
            return self.mlp(x).squeeze(-1)        # [K]


class GNNSimpleLP(nn.Module):
    """
    GNN sencilla para Link Prediction sobre snapshots de redes de pases.

    - encode(data) -> z: emb de nodos
    - score_pairs(z, edge_pairs) -> logits para cada arista (u->v)
    - score_edges(data, edge_pairs) -> encode + score_pairs
    - loss(data, pos, neg) -> BCEWithLogits sobre pos/neg
    """
    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        layers: int = 2,
        out: int = 64,
        dropout: float = 0.1,
        scorer: str = "bilinear",      # <-- cambio rentable por defecto
        scorer_mlp_hidden: int = 64,
    ):
        super().__init__()
        assert layers >= 1
        self.layers = layers
        self.dropout_p = dropout

        # GNN backbone
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))     # primera capa
        for _ in range(layers - 1):                    # capas intermedias
            self.convs.append(GCNConv(hidden, hidden))

        self.proj = nn.Linear(hidden, out)             # proyección al espacio 'out'
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # scorer configurable (dot | bilinear | mlp)
        self.scorer = EdgeScorer(mode=scorer, dim=out, mlp_hidden=scorer_mlp_hidden, dropout=dropout)

    def encode(self, data):
        """
        data: torch_geometric.data.Data con:
            - x: [N, in_dim]
            - edge_index: [2, E]
        """
        x, edge_index = data.x, data.edge_index
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)
        z = self.proj(h)  # [N, out]
        return z

    def score_pairs(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        """
        z: [N, out]
        edge_pairs: [2, K] con índices de nodos (u, v)
        return: logits [K]
        """
        return self.scorer(z, edge_pairs)

    @torch.no_grad()
    def score_edges(self, data, edge_pairs: torch.Tensor) -> torch.Tensor:
        """
        Atajo que hace encode + score_pairs, útil para evaluación.
        Devuelve logits (NO aplica sigmoide).
        """
        z = self.encode(data)
        return self.score_pairs(z, edge_pairs)

    def loss(self, data, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """
        BCE binaria con logits:
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

    def forward(self, data):
        # no se usa directamente para LP, pero se deja por compatibilidad
        return self.encode(data)