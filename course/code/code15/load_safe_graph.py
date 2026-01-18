# load_safe_graph.py
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import HeteroData

add_safe_globals([HeteroData])
graph = torch.load("simple_graph.pt", weights_only=False)
print(graph)

