from torch import nn
from abc import ABC, abstractmethod
from typing import *
from torch_geometric.data import Batch
from proteinvirtual.models.utils import get_gnn_layer

class AbstractGNNLayer(ABC):
    @abstractmethod
    def has_skip_connection(x) -> bool:
        pass
    
    @abstractmethod
    def upwards(self, input_batch) ->  Union[Batch, Tuple[Batch, Batch]]:
        pass
    
    @abstractmethod
    def downwards(self, input_batch, sc_batch = None) -> Batch:
        pass 


class HierarchicalGNNLayer(nn.Module):
    def __init__(self,
                 intra_conv_name: str,
                 up_conv_name: Optional[str],
                 down_conv_name: Optional[str],
                 emb_dim: int,
                 aggr: str = "mean",
                 dropout: float = 0.1,
                 skip_connect_type: Optional[str] = "sum"):
        super().__init__()
        self.intra_conv = get_gnn_layer(layer_name=intra_conv_name, emb_dim=emb_dim, aggr=aggr)
        self.is_final_layer = True
        if up_conv_name:
            self.up_conv = get_gnn_layer(layer_name=up_conv_name, emb_dim=emb_dim, aggr=aggr)
            self.is_final_layer = False 
        if down_conv_name:
            self.down_conv = get_gnn_layer(layer_name=down_conv_name, emb_dim=emb_dim, aggr=aggr)
            self.is_final_layer = False  # Final layer does not need up/down conv
        if skip_connect_type is None:
            self.skip_connect_type = "none"
        else:
            self.skip_connect_type = skip_connect_type.strip().lower()
        self.dropout = nn.DropOut(dropout)
        self.activation = nn.ReLU()
        match self.skip_connect_type:
            case "sum" | "mean" | "none":
                pass
            case "weighted":
                self.register_parameter("sc_weight", nn.Parameter(1.0, requires_grad=True))
            case _:
                raise ValueError(f"Invalid skip_connect_type: {self.skip_connect_type}")
    
    def _combine_skip_connect(self,
                              x_old,
                              x_new):
        match self.skip_connect_type:
            case "sum":
                return x_old + x_new
            case "mean":
                return (x_old + x_new)/2.0
            case "none":
                return x_new
            case "weighted":
                return (x_old + x_new * self.sc_weight)/(1.0+self.sc_weight)
            case _:
                raise ValueError(f"Invalid skip_connect_type: {self.skip_connect_type}")
            
    def upwards(self,
                x_lower,
                x_this,
                edge_index,
                **kwargs):
        x = x_this + self.up_conv(x=(x_lower, x_this), edge_index=edge_index, **kwargs)
        x = self.activation(self.dropout(x))
        return x
    
    def conv(self,
             x,
             edge_index,
             **kwargs):
        x_update = self.intra_conv(x, edge_index=edge_index, **kwargs)
        x = self.activation(self.dropout(x + x_update))
        return x
    
    def downwards(self,
                  x_upper,
                  x_this,
                  edge_index,
                  **kwargs):
        x_from_upper = self.down_conv(x=(x_upper, x_this), edge_index=edge_index, **kwargs)
        x = self._combine_skip_connect(x_this, x_from_upper)
        x = self.activation(self.dropout(x))
        return x
        
        
class HierarchicalGNN(nn.Module):
    def __init__(self,
                 layers: Iterable[HierarchicalGNNLayer],
                 virtual_layer_prefix="virtual",
                 inter_edge_name="inter",
                 intra_edge_name="intra"):
        super().__init__()
        self.layers = list(layers)
        
    @classmethod
    def construct(cls,
                  layer_specs, 
                  **kwargs):
        layers = []
        for layer_spec in layer_specs:
            if "_repeat_" in layer_spec.keys():
                repeat = int(layer_spec['_repeat_'])
                del layer_spec['_repeat_']
            else:
                repeat = 1
            
            for _ in range(repeat):
                layers.append(HierarchicalGNNLayer(**layer_spec))
        return cls(layers=layers, **kwargs)
                
    def forward(self,
                batch):
        stack = []
        # Upwards
        for i_layer in range(self.num_layers()):
            getattr()
        
        # Downwards
        
        pass
    
    def num_layers(self) -> int:
        return len(self.layers)