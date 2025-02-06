from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class WeightedPool(nn.Module):
    def __init__(self, 
                 emb_dim: int, 
                 normalise: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.normalise = normalise
        self.weight_model = nn.Linear(emb_dim, 1)
    
    def forward(self, x, batch):
        w = F.sigmoid(self.weight_model(x))
        wx = w * x
        out = scatter(wx, batch, dim=-2, reduce="sum")
        if self.normalise:
            w_batch = scatter(w, batch, dim=0, reduce="sum")
            return out/w_batch
        else:
            return out


def adaptive_conv(conv,
                  node_features,
                  edge_features,
                  supports_edge_weight: bool = False,
                  supports_edge_attr: bool = False,
                  supports_pos: bool = False):
    if isinstance(node_features, tuple):
        x = node_features[0].x, node_features[1].x
        pos = getattr(node_features[0], "pos", None), getattr(node_features[1], "pos", None)
    else:
        x = node_features.x
        pos = getattr(node_features, "pos", None)
    edge_index = edge_features.edge_index
    edge_weight = getattr(edge_features, "edge_weight", None)
    edge_attr = getattr(edge_features, "edge_attr", None)

    if supports_pos:
        if supports_edge_weight and supports_edge_attr:
            out = conv(x, edge_index, pos=pos, edge_weight=edge_weight,
                        edge_attr=edge_attr)
        elif supports_edge_weight:
            out = conv(x, edge_index, pos=pos, edge_weight=edge_weight)
        elif supports_edge_attr:
            out = conv(x, edge_index, pos=pos, edge_attr=edge_attr)
        else:
            out = conv(x, edge_index, pos=pos)
    else:
        if supports_edge_weight and supports_edge_attr:
            out = conv(x, edge_index, edge_weight=edge_weight,
                        edge_attr=edge_attr)
        elif supports_edge_weight:
            out = conv(x, edge_index, edge_weight=edge_weight)
        elif supports_edge_attr:
            out = conv(x, edge_index, edge_attr=edge_attr)
        else:
            out = conv(x, edge_index)
    return out