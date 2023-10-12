import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b

    正则, 归一化，dot乘积
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super(MetricCalcLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)
        self.lin = nn.Linear(nhid, nhid)
        self.Bi = nn.Bilinear(nhid, nhid, 1)

    def forward(self, h):
        return h * self.weight
        # return self.lin(h)


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape)  # .cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask.cuda()
    return sparse_graph


class GraphGenerator(nn.Module):
    """
    Generate graph using similarity.
    """

    def __init__(self, dim, num_head=1, threshold=0.2, dev=None, reverse=False):  # threshold=0.1
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
            # self.metric_layer.append(nn.Bilinear(dim, dim, 1))
        self.num_head = num_head
        self.dev = dev
        self.reverse = reverse

    def forward(self, x, K):
        K = int((x.shape[0]*K))
        sim = torch.zeros((x.shape[0], x.shape[0])).to('cuda')
        '''多头求和 {cos(W_i * h1, W_i * h2)}'''
        for i in range(self.num_head):
            w_x = self.metric_layer[i](x).cuda()
            sim += cos_sim(w_x, w_x)
        sim /= self.num_head
        # sim += x @ x.T
        # sim = torch.clamp(sim, 0.01, 0.99)
        # sim = RelaxedBernoulli(temperature=torch.Tensor([2.0]).to(sim.device), probs=sim).rsample()
        # 过滤掉相似度低的边, 即判断赋0
        if self.reverse == False:
            sim = torch.where(sim < self.threshold, torch.zeros_like(sim), sim)
        else:
            sim = torch.where(sim > self.threshold, torch.zeros_like(sim), sim)

        if K > 0:
            return top_k(sim, K)
        else:
            return torch.zeros_like(sim)#sim


class ViewLearner(torch.nn.Module):
    def __init__(self, input_dim, mlp_edge_model_dim=64):
        super(ViewLearner, self).__init__()
        self.input_dim = input_dim
        self.mlp_edge_model = Sequential(
            Linear(self.input_dim * 2, mlp_edge_model_dim),
            ReLU(),
            Linear(mlp_edge_model_dim, 1))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, node_emb, edge_index, dense=False):
        '''输入学习到的节点表示，按照已有的边进行edge dropping
            边的两端节点拼接后，执行一个MLP 输出的dim=1，---> 则计算出边对应的w'''

        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)
        '''
        relax p_e 到[0,1]的连续变量
            然后使用Gumbel-Max reparametrization trick

        torch.rand 从[0, 1]的均匀分布中抽取的一组随机数, eps 是随机的
        加bias 是让 eps 的范围 改为 (0, 1)

        eps 是 (0, 1)的正态分布
           p_e = sigmoid((log(eps) - log(1-eps) + w_e)/temp)

        F.gumbel_softmax 是torch里写好的程序，与论文代码差距在：
        gumbel_softmax 的输出使用了 softmax
        而论文的 Gumbel-Max 的输出使用了 sigmoid   
        '''
        temperature = 2.0  # 1.0
        # Gumbel-Max
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to('cuda')
        gate_inputs = (gate_inputs + edge_logits) / temperature
        edge_mask = torch.sigmoid(gate_inputs).squeeze().detach()
        # edge_mask = torch.softmax(gate_inputs, dim=1).squeeze().detach()
        # gumbel_softmax
        # edge_mask = F.gumbel_softmax(edge_logits)  # 高现存占用率
        edge_mask = torch.where(edge_mask < 0.0, torch.zeros_like(edge_mask), edge_mask)  # 0.3
        if dense == True:
            node_num = node_emb.size(0)
            g = torch.sparse.FloatTensor(edge_index, edge_mask, (node_num, node_num))
            g = g.to_dense()
            return g
        else:
            return edge_mask

class ViewLearner2(torch.nn.Module):
    def __init__(self, input_dim, mlp_edge_model_dim=64):
        super(ViewLearner2, self).__init__()
        self.input_dim = input_dim
        self.mlp_edge_model = Sequential(
            Linear(self.input_dim, mlp_edge_model_dim),
            ReLU(),
            Linear(mlp_edge_model_dim, 1))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, node_emb, edge_index, dense=False):
        '''输入学习到的节点表示，按照已有的边进行edge dropping
            边的两端节点拼接后，执行一个MLP 输出的dim=1，---> 则计算出边对应的w'''


        sim = cos_sim(node_emb, node_emb)
        edge_logits = self.mlp_edge_model(sim)

        temperature = 2.0  # 1.0
        # Gumbel-Max
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to('cuda')
        gate_inputs = (gate_inputs + edge_logits) / temperature
        edge_mask = torch.sigmoid(gate_inputs).squeeze().detach()
        # edge_mask = torch.softmax(gate_inputs, dim=1).squeeze().detach()
        # gumbel_softmax
        # edge_mask = F.gumbel_softmax(edge_logits)  # 高现存占用率
        edge_mask = torch.where(edge_mask < 0.0, torch.zeros_like(edge_mask), edge_mask)  # 0.3
        if dense == True:
            node_num = node_emb.size(0)
            g = torch.sparse.FloatTensor(edge_index, edge_mask, (node_num, node_num))
            g = g.to_dense()
            return g
        else:
            return edge_mask