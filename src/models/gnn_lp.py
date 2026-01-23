# src/models/gnn_lp.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class EdgeScorer(nn.Module):
    """
    Puntúa pares (u,v) a partir de embeddings de nodos.
    modes:
      - "dot":       producto punto (baseline)
      - "bilinear":  capa bilinear
      - "mlp":       concat(z_u,z_v) -> MLP
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
            return (zu * zv).sum(dim=-1)  # [K]
        elif self.mode == "bilinear":
            return self.bilin(zu, zv).squeeze(-1)  # [K]
        else:
            x = torch.cat([zu, zv], dim=-1)       # [K, 2D]
            return self.mlp(x).squeeze(-1)        # [K]


class GNNSimpleLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        layers: int = 2,
        out: int = 64,
        dropout: float = 0.1,
        scorer: str = "bilinear",
        scorer_mlp_hidden: int = 64,
        conv_type: str = "gcn",         # <-- añade este parámetro
        gat_heads: int = 4,             # <-- para GAT
    ):
        super().__init__()
        assert layers >= 1
        self.layers = layers
        self.dropout_p = dropout
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        conv_type = (conv_type or "gcn").lower()
        self.convs = nn.ModuleList()

        if conv_type == "gcn":
            self.convs.append(GCNConv(in_dim, hidden))
            for _ in range(layers - 1):
                self.convs.append(GCNConv(hidden, hidden))

        elif conv_type == "sage":
            self.convs.append(SAGEConv(in_dim, hidden))
            for _ in range(layers - 1):
                self.convs.append(SAGEConv(hidden, hidden))

        elif conv_type == "gat":
            self.convs.append(GATConv(in_dim, hidden, heads=gat_heads, concat=False, dropout=dropout))
            for _ in range(layers - 1):
                self.convs.append(GATConv(hidden, hidden, heads=gat_heads, concat=False, dropout=dropout))

        else:
            raise ValueError(f"conv_type desconocido: {conv_type}")

        self.conv_type = conv_type
        self.proj = nn.Linear(hidden, out)
        self.scorer = EdgeScorer(mode=scorer, dim=out, mlp_hidden=scorer_mlp_hidden, dropout=dropout)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)
        z = self.proj(h)
        return z

    def score_pairs(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        return self.scorer(z, edge_pairs)

    @torch.no_grad()
    def score_edges(self, data, edge_pairs: torch.Tensor) -> torch.Tensor:
        z = self.encode(data)
        return self.score_pairs(z, edge_pairs)

    def loss(self, data, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        z = self.encode(data)
        pos_logits = self.score_pairs(z, pos)
        neg_logits = self.score_pairs(z, neg)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        return nn.BCEWithLogitsLoss()(logits, labels)

    def forward(self, data):
        return self.encode(data)
