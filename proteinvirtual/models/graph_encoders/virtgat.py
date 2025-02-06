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
from proteinvirtual.models.utils import WeightedPool, adaptive_conv


class RealGAT(GAT):
    def __init__(self, pool: str = "mean", **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.pool = get_aggregation(pool)
        
    @property
    def required_batch_attributes(self) -> Set[str]:
        return {}
        
    def forward(self,
                batch):
        node_features = batch['real']
        edge_features = batch['real', 'r_to_r', 'real']
        x = node_features.x
        edge_index = edge_features.edge_index
        edge_weight = getattr(edge_features, "edge_weight", None)
        edge_attr = getattr(edge_features, "edge_attr", None)
        x = super(GAT, self).forward(
            x, edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=node_features.batch,
        )
        return EncoderOutput(
            {
                "node_embedding": x,
                "graph_embedding": self.pool(
                    x, node_features.batch
                )
            }
        )


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
        max_levels: int = 1,
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
        self.max_levels = max_levels
        self.convs_up =    ModuleList(ModuleList() for _ in range(num_layers-1))
        self.convs_down =  ModuleList(ModuleList() for _ in range(num_layers-1))
        self.convs_intra = ModuleList(ModuleList() for _ in range(num_layers-1))
        self.norms_up =    ModuleList(ModuleList() for _ in range(num_layers-1))
        self.norms_down =  ModuleList(ModuleList() for _ in range(num_layers-1))
        self.norms_intra = ModuleList(ModuleList() for _ in range(num_layers-1))
        self.virtual_in_channels = virtual_in_channels
        if pool.strip().lower() == "weighted":
            self.pool = WeightedPool(emb_dim=out_channels)
        else:
            self.pool = get_aggregation(pool)

        if isinstance(in_channels, tuple):
            raise Exception("in_channel cannot be a tuple: bipartite input not supported.")

        norm_layer = normalization_resolver(
            norm,
            hidden_channels,
            **(norm_kwargs or {}),
        ) or torch.nn.Identity()

        for idx_layer in range(num_layers-1):
            if idx_layer > 0:
                virtual_in_channels = hidden_channels
            for idx_level in range(max_levels):
                self.convs_up[idx_layer].append(self.init_conv(
                    in_channels=(hidden_channels, virtual_in_channels),
                    out_channels=hidden_channels,
                    **kwargs
                ))

                self.convs_intra[idx_layer].append(self.init_conv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    **kwargs
                ))

                self.convs_down[idx_layer].append(self.init_conv(
                    in_channels=(hidden_channels, hidden_channels),
                    out_channels=hidden_channels,
                    **kwargs
                ))
                self.norms_up[idx_layer].append(deepcopy(norm_layer))
                self.norms_intra[idx_layer].append(deepcopy(norm_layer))
                self.norms_down[idx_layer].append(deepcopy(norm_layer))

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {}
    
    def forward(self,
                batch: Union[Batch, ProteinBatch]):
        xs: List[Tensor] = []
        for idx_layer in range(self.num_layers):
            # convolution: real nodes
            node_features = batch[self.__rnode_name]
            edge_name = self.__rnode_name, self.__redge_name, self.__rnode_name
            edge_features = batch[edge_name]
            delta = adaptive_conv(self.convs[idx_layer],
                            node_features,
                            edge_features,
                            supports_edge_weight=False,
                            supports_edge_attr=True,
                            supports_pos=False)
            if not isinstance(self.norms[idx_layer], torch.nn.Identity):
                delta = self.norms[idx_layer](delta,
                                              batch=node_features.batch)
            if self.convs[idx_layer].in_channels == self.convs[idx_layer].out_channels:
                x = x + delta
            else:
                x = delta
            x = self.act(x)
            if idx_layer < self.num_layers-1:
                # x = self.norms[idx_layer](x, node_features.batch, batch.batch_size)
                batch[self.__rnode_name].x = x
                prev_node_name = self.__rnode_name
                max_level = -1
                # Up the hierarchy
                for idx_level in range(self.max_levels):
                    node_name = self.__vnode_prefix+str(idx_level)
                    if node_name not in batch.node_types:
                        break

                    vnode_features = batch[prev_node_name], batch[node_name]
                    edge_name = prev_node_name, self.__up_vedge_name, node_name
                    vedge_features = batch[edge_name]
                    delta = adaptive_conv(self.convs_up[idx_layer][idx_level],
                                    vnode_features,
                                    vedge_features,
                                    supports_edge_weight=False,
                                    supports_edge_attr=True,
                                    supports_pos=False)
                    delta = self.norms_up[idx_layer][idx_level](delta, 
                                                                batch=batch[node_name].batch)
                    if idx_layer == 0:
                        x = delta
                    else:
                        x = batch[node_name].x + delta
                    x = self.act(x)

                    vnode_features = batch[node_name]
                    vnode_features.x = x
                    edge_name = node_name, self.__intra_vedge_name, node_name
                    vedge_features = batch[edge_name]
                    delta = adaptive_conv(self.convs_intra[idx_layer][idx_level],
                                          vnode_features,
                                          vedge_features,
                                          supports_edge_weight=False,
                                          supports_edge_attr=True,
                                          supports_pos=False)
                    delta = self.norms_intra[idx_layer][idx_level](delta,
                                                                   batch=batch[node_name].batch)
                    x = x + delta
                    x = self.act(x)
                    batch[node_name].x = x
                    prev_node_name = node_name
                    max_level = idx_level

                # Down the hierarchy
                for idx_level in range(max_level, -1, -1):
                    next_node_name = self.__vnode_prefix+str(idx_level-1) if idx_level > 0 else self.__rnode_name
                    node_name = self.__vnode_prefix+str(idx_level)
                    vnode_features = batch[node_name], batch[next_node_name]
                    edge_name = node_name, self.__down_vedge_name, next_node_name
                    vedge_features = batch[edge_name]
                    delta = adaptive_conv(self.convs_down[idx_layer][idx_level],
                                    vnode_features,
                                    vedge_features,
                                    supports_edge_weight=False,
                                    supports_edge_attr=True,
                                    supports_pos=False)
                    delta = self.norms_down[idx_layer][idx_level](delta,
                                                                  batch=batch[next_node_name].batch)
                    if idx_layer == self.num_layers-1:
                        x = delta
                    else:
                        x = batch[next_node_name].x + delta
                    x = self.act(x)
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
