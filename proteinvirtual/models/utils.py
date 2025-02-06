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

