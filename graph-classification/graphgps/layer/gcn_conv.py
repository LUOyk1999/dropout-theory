import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_scatter import scatter
import torch
import torch.nn.functional as F

import ipdb
from torch_geometric.utils import degree
from torch_geometric.utils import scatter

def dirichlet_energy(x, edge_index, batch=None):
    
    with torch.no_grad():
        src, dst = edge_index
        deg = degree(src, num_nodes=x.shape[0])

        x = x / torch.sqrt(deg + 1.0).view(-1, 1)
        energy = torch.norm(x[src] - x[dst], dim=1, p=2) ** 2.0
        # ipdb.set_trace()
        if batch is not None:
            energy = scatter(energy, batch[dst], dim_size=x.shape[0], reduce='mean')[:batch[-1]]
        else:
            energy = energy.mean()

        energy *= 0.5

    return float(energy.mean().detach().cpu())

class GCNConvLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = False
        self.batch_norm = True
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_in)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_in)

        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        self.model = pyg_nn.GCNConv(dim_in, dim_out, bias=True)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index)
        if self.layer_norm:
            batch.x = self.norm1_local(batch.x, batch.batch)
        if self.batch_norm:
            batch.x = self.norm1_local(batch.x)
        batch.x = self.act(batch.x)
        if self.residual:
            batch.x = x_in + batch.x  # residual connection
        
        return batch
