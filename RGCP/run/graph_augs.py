import numpy as np
from IPython import embed
import copy
import pandas as pd
# from matplotlib import pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   to_undirected)
from torch_sparse import SparseTensor
from torch_geometric.utils import subgraph, k_hop_subgraph

def NodeDrop(data, aug_ratio):
    data = copy.deepcopy(data)
    x = data.x#[:, :300]
    edge_index = data.edge_index
    # 随机dropout
    drop_num = int(data.num_nodes * aug_ratio)
    keep_num = data.num_nodes - drop_num
    keep_idx = torch.randperm(data.num_nodes)[:keep_num]
    edge_index, _ = subgraph(keep_idx, edge_index)
    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[keep_idx] = False
    # root 不 dropout
    drop_idx[data.root_index] = False
    x[drop_idx] = 0
    data.x = x
    data.edge_index = edge_index
    return data

def EdgePerturb(data, aug_ratio):
    data = copy.deepcopy(data)
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index

    unif = torch.ones(2, node_num)
    add_edge_idx = unif.multinomial(permute_num, replacement=True).to(data.x.device)

    # # 随机抽样
    unif = torch.ones(edge_num)
    keep_edge_idx = unif.multinomial((edge_num - permute_num), replacement=False)

    # edge_index = edge_index[:, keep_edge_idx]
    #
    edge_index = torch.cat((edge_index[:, keep_edge_idx], add_edge_idx), dim=1)
    data.edge_index = edge_index
    return data

def AttrMask(data, aug_ratio):
    data = copy.deepcopy(data)
    # 随机 mask
    mask_num = int(data.num_nodes * aug_ratio)
    unif = torch.ones(data.num_nodes)
    unif[data.root_index] = 0
    mask_idx = unif.multinomial(mask_num, replacement=False)
    token = data.x.mean(dim=0)
    # shape = data.x.mean(dim=0).shape
    # token = torch.rand(shape[0]).cuda()
    data.x[mask_idx] = token
    return data

class Graph_Augmentor(nn.Module):
    def __init__(self, aug_ratio, preset=-1):
        super().__init__()
        self.aug_ratio = aug_ratio
        self.aug = preset
    
    def forward(self, data):
        data1 = data
        data = copy.deepcopy(data)
        if self.aug_ratio > 0:
            self.aug = 1#np.random.randint(3)
            if self.aug == 0:
                # print("node drop")
                data = NodeDrop(data, self.aug_ratio)
            elif self.aug == 1:
                # print("edge perturb")
                data = EdgePerturb(data, self.aug_ratio)
            elif self.aug == 2:
                # print("attr mask")
                data = AttrMask(data, self.aug_ratio)
            elif self.aug == 3:
                # print("attr mask")
                data = AttrMask(data, self.aug_ratio)
                data = NodeDrop(data, self.aug_ratio)
                data = EdgePerturb(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False
        return data

def edgemask_um(mask_ratio, edge_index, device, num_nodes):
    num_edge = edge_index.shape[1]#len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num]).type(torch.long).cuda()
    mask_index = torch.from_numpy(index[-mask_num:]).type(torch.long).cuda()
    edge_index_train = edge_index[:, pre_index]#.t()
    edge_index_mask = edge_index[:, mask_index]#.t()
    edge_index = to_undirected(edge_index_train)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index)#.t()
    return adj, edge_index, edge_index_mask.to(device)


def edgemask_dm(mask_ratio, split_edge, device, num_nodes):
    # if isinstance(split_edge, torch.Tensor):
    #     edge_index = to_undirected(split_edge.t()).t()
    # else:
    #     edge_index = torch.stack([split_edge['train']['edge'][:, 1], split_edge['train']['edge'][:, 0]], dim=1)
    #     edge_index = torch.cat([split_edge['train']['edge'], edge_index], dim=0)
    edge_index = split_edge#to_undirected(split_edge)
    num_edge = edge_index.shape[1]#len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)  # 需要mask的边数量
    pre_index = torch.from_numpy(index[0:-mask_num]).type(torch.long).cuda() # mask后保留的edge_index
    mask_index = torch.from_numpy(index[-mask_num:]).type(torch.long).cuda() # mask 的 edge_index
    edge_index_train = edge_index[:, pre_index]  # 用于训练
    edge_index_mask = edge_index[:, mask_index]  # 被mask的

    edge_index = edge_index_train
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index)#.t()
    return adj, edge_index, edge_index_mask.to(device)

def mask_edge(edge_index, p, weight=None):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    if weight == None:
        return edge_index[:, ~mask], edge_index[:, mask]
    else:
        return edge_index[:, ~mask], edge_index[:, mask], weight[~mask], weight[mask]

def encoding_mask_noise(x, mask_rate=0.3, replace_rate=0.0):
    num_nodes = x.shape[0]
    # enc_mask_token = nn.Parameter(torch.zeros(1, x.shape[1])).cuda()
    perm = torch.randperm(num_nodes, device=x.device)
    # random masking
    num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]
    keep_nodes = perm[num_mask_nodes: ]

    if replace_rate > 0:
        num_noise_nodes = int(replace_rate * num_mask_nodes)
        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        token_nodes = mask_nodes[perm_mask[: int(replace_rate * num_mask_nodes)]]
        noise_nodes = mask_nodes[perm_mask[-int(replace_rate * num_mask_nodes):]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[noise_nodes] = x[noise_to_be_chosen]
    else:
        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

    # out_x[token_nodes] += enc_mask_token
    return out_x, (mask_nodes, keep_nodes)

class Learned_mask_edge(torch.nn.Module):
    def __init__(self, input_dim, mlp_edge_model_dim=64):
        super(Learned_mask_edge, self).__init__()
        self.input_dim = input_dim
        self.mlp_edge_model = nn.Sequential(
            nn.Linear(self.input_dim*2, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, node_emb, edge_index, mask_rate, weight, probability=None):
        '''输入学习到的节点表示，按照已有的边进行edge dropping
            边的两端节点拼接后，执行一个MLP 输出的dim=1，---> 则计算出边对应的w'''
        src, dst = edge_index[0], edge_index[1]
        if probability == None:

            emb_src = node_emb[src]
            emb_dst = node_emb[dst]

            edge_emb = torch.cat([emb_src, emb_dst], 1)
            edge_logits = self.mlp_edge_model(edge_emb)
        else:
            edge_logits = probability[src, dst].unsqueeze(1)

        temperature = 1  # 1.0
        # Gumbel-Max
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(node_emb.device)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

        edge_mask_prob = 1 - edge_weight  # [edge_num]
        mask_num = int(edge_weight.shape[0] * mask_rate)
        mask_idx = edge_mask_prob.multinomial(mask_num, replacement=False)

        if weight == None:
            return edge_index[:, ~mask_idx], edge_index[:, mask_idx]
        else:
            return edge_index[:, ~mask_idx], edge_index[:, mask_idx], weight[~mask_idx], weight[mask_idx]


class Learned_drop_node(torch.nn.Module):
    def __init__(self, input_dim, mlp_edge_model_dim=64):
        super(Learned_drop_node, self).__init__()
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, node_emb, mask_rate):
        '''输入学习到的节点表示，按照已有的边进行edge dropping
            边的两端节点拼接后，执行一个MLP 输出的dim=1，---> 则计算出边对应的w'''

        node_logits = self.mlp(node_emb)
        temperature = 1 # 1.0
        # Gumbel-Max
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(node_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(node_emb.device)
        gate_inputs = (gate_inputs + node_logits) / temperature
        node_weight = torch.sigmoid(gate_inputs).squeeze().detach()

        edge_mask_prob = 1 - node_weight  # [edge_num]
        mask_num = int(node_emb.shape[0] * mask_rate)
        mask_idx = edge_mask_prob.multinomial(mask_num, replacement=False)

        out_x = node_emb.clone()
        out_x[mask_idx] = 0.0

        return out_x, (mask_idx, 0)
