from typing import *
from copy import deepcopy

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn.models import GAT
from torch_geometric.nn.conv import gat_conv, gatv2_conv
from torch_geometric.data import Batch
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.nn.resolver import normalization_resolver
from proteinworkshop.types import EncoderOutput
from proteinworkshop.models.utils import get_aggregation


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


class VirtualGAT(GAT):
    __rnode_name = "real"
    __redge_name = "r_to_r"
    __vnode_prefix = "vnode_"
    __intra_vedge_name = "intra"
    __up_vedge_name = "inter"
    __down_vedge_name = "inter"
    def __init__(
        self, 
        in_channels: int,
        virtual_in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        v2: bool = None,
        max_hierarchies: int = 1,
        pool: str = "mean",
        **kwargs
    ):
        super(GAT, self).__init__(
                         in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=out_channels,
                         v2=v2,
                         dropout=dropout,
                         act=act,
                         act_first=act_first,
                         act_kwargs=act_kwargs,
                         norm=norm,
                         norm_kwargs=norm_kwargs,
                         jk=jk,
                         **kwargs)
        self.num_layers = num_layers
        self.max_hierarchies = max_hierarchies
        self.convs_up = ModuleList(ModuleList() for _ in range(num_layers))
        self.convs_down = ModuleList(ModuleList() for _ in range(num_layers))
        self.convs_intra = ModuleList(ModuleList() for _ in range(num_layers))
        self.norms_up = ModuleList(ModuleList() for _ in range(num_layers))
        self.norms_down = ModuleList(ModuleList() for _ in range(num_layers))
        self.norms_intra = ModuleList(ModuleList() for _ in range(num_layers))
        self.virtual_in_channels = virtual_in_channels
        self.pool = get_aggregation(pool)
        
        if isinstance(in_channels, tuple):
            raise Exception("in_channel cannot be a tuple: bipartite input not supported.")
        
        norm_layer = normalization_resolver(
            norm,
            hidden_channels,
            **(norm_kwargs or {}),
        )
        if norm_layer is None:
            norm_layer = torch.nn.Identity()
        
        for idx_layer in range(num_layers):
            if idx_layer > 0:
                virtual_in_channels = hidden_channels
            for idx_hierarchy in range(max_hierarchies-1):
                self.convs_up[idx_layer].append(self.init_conv(
                    in_channels=(hidden_channels, virtual_in_channels),
                    out_channels=hidden_channels,
                    **kwargs
                ))
                # self.norms_up[idx_layer].append(deepcopy(norm_layer))
                
                self.convs_intra[idx_layer].append(self.init_conv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    **kwargs
                ))
                # self.norms_intra.append(deepcopy(norm_layer))
                
                self.convs_down[idx_layer].append(self.init_conv(
                    in_channels=(hidden_channels, hidden_channels),
                    out_channels=hidden_channels if idx_layer == num_layers-1 else out_channels,
                    **kwargs
                ))
                # self.norms_down[idx_layer].append(deepcopy(norm_layer))
        
            # # Final hierarchy
            # self.convs_intra[idx_layer].append(self.init_conv(
            #     in_channels=hidden_channels, 
            #     out_channels=hidden_channels,
            #     **kwargs
            # ))
            # self.norms_intra.append(deepcopy(norm_layer))
    
    @property
    def required_batch_attributes(self) -> Set[str]:
        return {}
    
    def forward(self,
                batch: Union[Batch, ProteinBatch]):
        xs: List[Tensor] = []
        for idx_layer in range(self.num_layers):
            skip_connect_stack = []
            # convolution: real nodes
            node_features = batch[self.__rnode_name]
            edge_name = self.__rnode_name, self.__redge_name, self.__rnode_name
            edge_features = batch[edge_name]
            x = adaptive_conv(self.convs[idx_layer],
                              node_features,
                              edge_features,
                              supports_edge_weight=False,
                              supports_edge_attr=True,
                              supports_pos=False)
            x = self.act(x)
            # x = self.norms[idx_layer](x, node_features.batch, batch.batch_size)
            batch[self.__rnode_name].x = x
            prev_node_name = self.__rnode_name
            max_hierach = -1
            # Up the hierarchy
            for idx_hierach in range(self.max_hierarchies):
                node_name = self.__vnode_prefix+str(idx_hierach)
                if not hasattr(batch, node_name):
                    break
                
                vnode_features = batch[prev_node_name], batch[node_name]
                edge_name = prev_node_name, self.__up_vedge_name, node_name
                vedge_features = batch[edge_name]
                x = adaptive_conv(self.convs_up[idx_layer][idx_hierach],
                                  vnode_features,
                                  vedge_features,
                                  supports_edge_weight=False,
                                  supports_edge_attr=True,
                                  supports_pos=False)
                x = self.act(x)
                # x = self.norms_up[idx_layer][idx_hierach](
                #     x, node_features[1].batch,
                #     batch.batch_size
                # )
                
                vnode_features = batch[node_name]
                vnode_features.x = x
                edge_name = node_name, self.__intra_vedge_name, node_name
                vedge_features = batch[edge_name]
                x = adaptive_conv(self.convs_intra[idx_layer][idx_hierach],
                                  vnode_features,
                                  vedge_features,
                                  supports_edge_weight=False,
                                  supports_edge_attr=True,
                                  supports_pos=False)
                x = self.act(x)
                # x = self.norms_intra[idx_layer][idx_hierach](
                #     x, node_features.batch,
                #     batch.batch_size
                # )
                batch[node_name].x = x
                prev_node_name = node_name
                max_hierach = idx_hierach
            
            # Down the hierarchy
            for idx_hierach in range(max_hierach, -1, -1):
                next_node_name = self.__vnode_prefix+str(idx_hierach-1) if idx_hierach > 0 else self.__rnode_name
                node_name = self.__vnode_prefix+str(idx_hierach) 
                vnode_features = batch[node_name], batch[next_node_name]
                edge_name = node_name, self.__down_vedge_name, next_node_name
                vedge_features = batch[edge_name]
                x = adaptive_conv(self.convs_down[idx_layer][idx_hierach],
                                  vnode_features,
                                  vedge_features,
                                  supports_edge_weight=False,
                                  supports_edge_attr=True,
                                  supports_pos=False)
                x = self.act(x)
                # x = self.norms_intra[idx_layer][idx_hierach](
                #     x, node_features[0].batch,
                #     batch.batch_size
                # )
                batch[next_node_name].x = x
            
            # Wrap things up
            x = batch[self.__rnode_name].x
            x = self.dropout(x)
            if hasattr(self, 'jk'):
                    xs.append(x)
            
        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return EncoderOutput(
            {
                "node_embedding": x,
                "graph_embedding": self.pool(
                    x, batch[self.__rnode_name].batch
                )
            }
        )