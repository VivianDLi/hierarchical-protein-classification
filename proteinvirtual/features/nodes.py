import math
from typing import Mapping
import torch
from torch_scatter import scatter
from torch_geometric.data import HeteroData, Data
from torch_geometric.data.batch import Batch, DynamicInheritanceGetter


def fps_torch(pos: torch.tensor, k: int = 1, select: int = 0) -> torch.tensor:
    """_summary_

    Args:
        pos (torch.tensor): (N, 3) tensor of positions
        k (int): number of samples to select
        select (int, optional): Optional initial position to be sampled

    Return:
        torch.tensor: (k, 3) tensor of sampled positions
    
    """
    
    n = pos.shape[0]
    assert pos.shape[1] == 3, "Position tensor must be (N, 3)"
    assert k <= n, "Number of samples must be less than or equal to the number of positions"

    selected = torch.zeros(n, dtype=torch.bool)
    selected[select] = True
    
    distances = torch.pairwise_distance(pos, pos[selected])
    distances[selected] = - math.inf
    
    for i in range(k-1):
        new_sample = torch.argmax(distances)
        selected[new_sample] = True
        distances = torch.min(distances, torch.pairwise_distance(pos, pos[new_sample]))
        distances[selected] = - math.inf

    return pos[selected]


def fps_torch_batch(pos: torch.tensor, k: int = 1, select: int = 0):
    pass

def compute_random_uniform(num_nodes, basis_pos, basis_batch=None):
    if basis_batch is None:
        assert isinstance(num_nodes, int)
        max = torch.max(basis_pos, dim=0)
        min = torch.min(basis_pos, dim=0)
        return torch.rand(num_nodes, 3, dtype=basis_pos.dtype, device=basis_pos.device) * (max - min) + min
    else:
        max = scatter(src=basis_pos, index=basis_batch, dim=0, reduce='max')  # (B, 3)
        min = scatter(src=basis_pos, index=basis_batch, dim=0, reduce='min')  # (B, 3)
        num_elements = max.shape[0] * num_nodes if isinstance(num_nodes, int) else torch.sum(num_nodes).int()
        max = torch.repeat_interleave(max, num_nodes, dim=0)
        min = torch.repeat_interleave(min, num_nodes, dim=0)
        rand = torch.rand(num_elements, 3, dtype=basis_pos.dtype, device=basis_pos.device)
        return rand * (max - min) + min
    
    
def scatter_std(x, index, mean=None):
    if mean is None:
        mean = scatter(x, index, dim=0, reduce='mean')
    mean = torch.gather(mean, dim=0, index=index.unsqueeze(-1).repeat(1, 3))
    return torch.sqrt(scatter((x - mean) ** 2, index, dim=0, reduce='mean'))


def compute_random_normal(num_nodes, basis_pos, basis_batch=None):
    if basis_batch is None:
        assert isinstance(num_nodes, int)
        mean = torch.mean(basis_pos, dim=0)
        std = torch.std(basis_pos, dim=0)
        return torch.randn(num_nodes, 3, dtype=basis_pos.dtype, device=basis_pos.device) * std + mean
    else:
        mean = scatter(src=basis_pos, index=basis_batch, dim=0, reduce='mean')  # (B, 3)
        std = scatter_std(basis_pos, basis_batch, mean)  # (B, 3)
        num_elements = max.shape[0] * num_nodes if isinstance(num_nodes, int) else torch.sum(num_nodes).int()
        mean = torch.repeat_interleave(mean, num_nodes, dim=0)
        std = torch.repeat_interleave(std, num_nodes, dim=0)
        rand = torch.randn(num_elements, 3, dtype=basis_pos.dtype, device=basis_pos.device)
        return rand * std + mean


def compute_fps(num_nodes, basis_pos, basis_batch=None, bsz=None):
    if basis_batch is None:
        return fps_torch(basis_pos, k=num_nodes)
    else:
        bsz = bsz or torch.max(basis_batch).item() + 1
        if isinstance(num_nodes, int):
            num_nodes = torch.tensor([num_nodes] * bsz, device=basis_pos.device, dtype=torch.long)
        else:
            assert isinstance(num_nodes, torch.Tensor), "num_nodes must either be an int or torch.Tensor"
            assert num_nodes.shape[0] == bsz, "batch size must match"
        num_real_nodes = torch.bincount(basis_batch, minlength=bsz)
        num_nodes = torch.minimum(num_nodes, num_real_nodes)
        num_nodes_list = list(num_nodes.cpu().numpy())
        out = torch.empty((sum(num_nodes_list), 3), dtype=basis_pos.dtype, device=basis_pos.device)
        ptr = 0  # I love serial programming :)
        for i in range(bsz):
            if num_nodes_list[i] == 0:
                continue
            out[ptr: ptr+num_nodes_list[i]] = fps_torch(basis_pos[basis_batch == i], k=num_nodes_list[i])
            ptr += num_nodes_list[i]
        return out.view(-1, 3), num_nodes

   
def add_vnode_positions(position, n_nodes, node_name, data: HeteroData):
    if position in ['random', 'random_normal', 'random_uniform', 'fps']:
        position = {
            "type": position, 
            "basis": "real"
        }
    
    assert "type" in position, "Position strategy must be specified"
    assert "basis" in position, "Basis node type must be specified"
    assert position["basis"] in data.node_types, f"Basis node type {position['basis']} must be present in the data"
    basis = position["basis"]
    assert hasattr(data[basis], "pos") and data[basis].pos is not None, f"Basis node positions must be present in the data"
    
    if position["type"] in ["random", "random_uniform"]:
        new_pos = compute_random_uniform(n_nodes, data[basis].pos)
    elif position["type"] == "random_normal":
        new_pos = compute_random_normal(n_nodes, data[basis].pos)
    elif position["type"] == "fps":
        new_pos = compute_fps(n_nodes, data[basis].pos)
    else:
        raise NotImplemented(f"Position strategy {position['type']} not implemented")
  
    data[node_name].pos = new_pos
    return data


def add_vnode_positions_batch(position,
                              n_nodes,
                              node_name,
                              batch: Batch) -> Batch:
    assert isinstance(batch, HeteroData), "Batch must be a Batch(HeteroData)"
    if position in ['random', 'random_normal', 'random_uniform', 'fps']:
        position = {
            "type": position, 
            "basis": "real"
        }
    assert "type" in position, "Position strategy must be specified"
    assert "basis" in position, "Basis node type must be specified"
    assert position["basis"] in batch.node_types, f"Basis node type {position['basis']} must be present in the data"
    basis = position["basis"]
    assert hasattr(batch[basis], "pos") and batch[basis].pos is not None, f"Basis node positions must be present in the data"
    assert hasattr(batch[basis], "batch") and batch[basis].batch is not None, f"Missing batch information for node type {basis}"
     
    if position["type"] in ["random", "random_uniform"]:
        new_pos = compute_random_uniform(n_nodes, batch[basis].pos, batch[basis].batch)
    elif position["type"] == "random_normal":
        new_pos = compute_random_normal(n_nodes, batch[basis].pos, batch[basis].batch)
    elif position["type"] == "fps":
        new_pos, n_nodes = compute_fps(n_nodes, batch[basis].pos, batch[basis].batch, bsz=batch.num_graphs)
    else:
        raise NotImplemented(f"Position strategy {position['type']} not implemented")
    
    new_batch = torch.arange(batch.num_graphs).to(dtype=torch.long, device=new_pos.device).repeat_interleave(n_nodes)
    batch[node_name].pos = new_pos
    batch[node_name].batch = new_batch
    return batch