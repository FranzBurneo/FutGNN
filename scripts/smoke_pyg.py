import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# grafo simple de 4 nodos con 3 features cada uno
x = torch.randn(4, 3)
edge_index = torch.tensor([[0, 1, 2, 0],
                           [1, 2, 0, 3]], dtype=torch.long)  # shape [2, E]

g = Data(x=x, edge_index=edge_index)

conv = GCNConv(in_channels=3, out_channels=2)
out = conv(g.x, g.edge_index)
print("Output shape:", out.shape)  # debe ser [4, 2]