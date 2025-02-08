from typing import *
from copy import deepcopy

import torch
from torch import Tensor
from torch.nn import ModuleList, Module, Dropout, Linear, Identity
from torch_geometric.nn.models import GAT
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch_geometric.data import Batch
from torch_geometric.nn.resolver import normalization_resolver, activation_resolver
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

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


class SimpleAggregationConv(Module):
    def __init__(self,
                 reduce: str = "mean"):
        super().__init__()
        self.reduce = reduce
    
    def forward(self,
                x: Tuple[Tensor], 
                edge_index: Tensor, 
                pos=None,
                edge_attr=None) -> Union[Tensor, Tuple[Tensor]]:
        x_from, x_to = x
        scattered = x_from[edge_index[0]]
        gathered_x = torch_scatter.scatter(scattered, edge_index[1], dim=0, reduce=self.reduce)
        if pos is not None:
            pos_from = pos[0]
            scattered_pos = pos_from[edge_index[0]]
            gathered_pos = torch_scatter.scatter(scattered_pos, edge_index[1], dim=0, reduce=self.reduce)
            return gathered_x, gathered_pos
        return gathered_x


class AttentionUnet(Module):
    def __init__(
        self,
        virtual_in_channels: int,
        hidden_channels: int,
        level_names: List[str],
        dropout: float = 0.0,
        simple_aggregation: Optional[str] = "mean",
        residual: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        v2: bool = None,
        **kwargs,
    ):
        super().__init__()
        self.level_names = level_names
        self.dropout =     Dropout(dropout)
        self.convs_up =    ModuleList()
        self.convs_down =  ModuleList()
        self.convs_intra = ModuleList()
        self.norms_up =    ModuleList()
        self.norms_down =  ModuleList()
        self.norms_intra = ModuleList()
        self.virtual_in_channels = virtual_in_channels
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.residual = residual
        self.simple_aggregation = simple_aggregation
        
        norm_layer = normalization_resolver(
            norm,
            hidden_channels,
            **(norm_kwargs or {}),
        ) or torch.nn.Identity()
        Conv = GATConv if not v2 else GATv2Conv
        
        if virtual_in_channels != hidden_channels:
            self.input_transform = Linear(virtual_in_channels, hidden_channels, bias=False)
        else:
            self.input_transform = Identity()
        
        for idx_level in range(len(level_names)-1):
            self.convs_intra.append(Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dropout=dropout,
                **kwargs
            ))
            if simple_aggregation is not None:
                self.convs_up.append(SimpleAggregationConv(simple_aggregation))
                self.convs_down.append(SimpleAggregationConv(simple_aggregation))
            else:
                self.convs_up.append(Conv(
                    in_channels=(hidden_channels, virtual_in_channels),
                    out_channels=hidden_channels,
                    dropout=dropout,
                    **kwargs
                ))
                self.convs_down.append(Conv(
                    in_channels=(hidden_channels, hidden_channels),
                    out_channels=hidden_channels,
                    dropout=dropout,
                    **kwargs
                ))
            self.norms_up.append(deepcopy(norm_layer))
            self.norms_intra.append(deepcopy(norm_layer))
            self.norms_down.append(deepcopy(norm_layer))
    
    @jaxtyped(typechecker=typechecker)
    def forward(self,
                batch: Union[Batch, ProteinBatch]) -> Union[Tensor, Tuple[Tensor]]:

        # Up the hierarchy
        for idx_level in range(len(self.level_names)-1):
            prev_node_name = self.level_names[idx_level]
            node_name = self.level_names[idx_level+1]
            if node_name not in batch.node_types:
                break

            vnode_features = batch[prev_node_name], batch[node_name]
            edge_name = prev_node_name, "inter", node_name
            vedge_features = batch[edge_name]
            delta = adaptive_conv(self.convs_up[idx_level],
                            vnode_features,
                            vedge_features,
                            supports_edge_weight=False,
                            supports_edge_attr=True,
                            supports_pos=False)
            delta = self.norms_up[idx_level](delta, batch=batch[node_name].batch)
            x = self.input_transform(batch[node_name].x)
            x = x + delta if self.residual else delta
            x = self.act(x)

            vnode_features = batch[node_name]
            vnode_features.x = x
            edge_name = node_name, "intra", node_name
            vedge_features = batch[edge_name]
            delta = adaptive_conv(self.convs_intra[idx_level],
                                    vnode_features,
                                    vedge_features,
                                    supports_edge_weight=False,
                                    supports_edge_attr=True,
                                    supports_pos=False)
            delta = self.norms_intra[idx_level](delta, batch=batch[node_name].batch)
            x = x + delta if self.residual else delta
            x = self.act(x)
            batch[node_name].x = x
            prev_node_name = node_name
            max_level = idx_level

        # Down the hierarchy
        for idx_level in range(max_level, -1, -1):
            next_node_name = self.level_names[idx_level]
            node_name = self.level_names[idx_level+1]
            vnode_features = batch[node_name], batch[next_node_name]
            edge_name = node_name, "inter", next_node_name
            vedge_features = batch[edge_name]
            delta = adaptive_conv(self.convs_down[idx_level],
                            vnode_features,
                            vedge_features,
                            supports_edge_weight=False,
                            supports_edge_attr=True,
                            supports_pos=False)
            delta = self.norms_down[idx_level](delta, batch=batch[next_node_name].batch)
            x = batch[next_node_name].x
            x = x + delta if self.residual else delta
            x = self.act(x)
            batch[next_node_name].x = x

        # Wrap things up
        batch[self.level_names[0]].x = x
        return x


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
        residual: bool = False,
        simple_aggregation: Optional[str] = "mean",
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
        self.unets = ModuleList()
        self.residual = residual
        self.simple_aggregation = simple_aggregation
        self.level_names = [self.__rnode_name] + [f"{self.__vnode_prefix}{idx}" for idx in range(max_levels)] 
        for idx_layer in range(num_layers-1):
            self.unets.append(
                AttentionUnet(virtual_in_channels=virtual_in_channels if idx_layer == 0 else hidden_channels,
                              hidden_channels=hidden_channels,
                              level_names=self.level_names,
                              dropout=dropout,
                              act=act,
                              act_kwargs=act_kwargs,
                              norm=norm,
                              norm_kwargs=norm_kwargs,
                              v2=v2,
                              residual=residual,
                              simple_aggregation=simple_aggregation,
                              **kwargs)
            )
        if pool.strip().lower() == "weighted":
            self.pool = WeightedPool(emb_dim=out_channels)
        else:
            self.pool = get_aggregation(pool)

        if isinstance(in_channels, tuple):
            raise Exception("in_channel cannot be a tuple: bipartite input not supported.")

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {}
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
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
            if not isinstance(self.norms[idx_layer], Identity):
                delta = self.norms[idx_layer](delta,
                                              batch=node_features.batch)
            if self.convs[idx_layer].in_channels == self.convs[idx_layer].out_channels:
                x = x + delta
            else:
                x = delta
            x = self.act(x)
            batch[self.__rnode_name].x = x
            if idx_layer < self.num_layers-1:
                x = self.unets[idx_layer](batch)

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
