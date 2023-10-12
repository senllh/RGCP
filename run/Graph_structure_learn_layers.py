import torch
from torch_geometric.nn import GCNConv, global_mean_pool, JumpingKnowledge
# from torch_geometric.utils import accuracy, to_dense_adj, dense_to_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli
import math
import copy

VERY_SMALL_NUMBER = 1e-12
INF = 1e20
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, graph_type, top_k=None, epsilon=None, num_pers=4, metric_type="attention",
                 feature_denoise=True, device=None):
        super(GraphLearner, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_pers = num_pers
        self.graph_type = graph_type
        self.top_k = top_k
        self.epsilon = epsilon
        self.metric_type = metric_type
        self.feature_denoise = feature_denoise

        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList([nn.Linear(self.input_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, -num_pers))
        elif metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, self.input_size)  # 这里num_pers 代表重复的次数
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))
        elif metric_type == 'weighted_cosine2':
            self.weight_tensor1 = torch.Tensor(num_pers, self.input_size)  # 这里num_pers 代表重复的次数
            self.weight_tensor1 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor1))
            self.weight_tensor2 = torch.Tensor(num_pers, self.input_size) # 这里num_pers 代表重复的次数
            self.weight_tensor2 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor2))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))
        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.leakyrelu = nn.LeakyReLU(0.2)
            print('[ GAT_Attention GraphLearner]')
        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        elif metric_type == 'transformer':
            self.linear_sim1 = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_sim2 = nn.Linear(input_size, hidden_size, bias=False)
        elif metric_type == 'cosine':
            pass
        elif metric_type == 'mlp':
            self.lin1 = nn.Linear(self.input_size, self.hidden_size)
            self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
        elif metric_type == 'multi_mlp':
            self.linear_sims1 = nn.ModuleList([nn.Linear(self.input_size, hidden_size, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(self.hidden_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))
        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(input_size, init_strategy="constant")

        print('[ Graph Learner metric type: {}, Graph Type: {} ]'.format(metric_type, self.graph_type))

    def reset_parameters(self):
        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(self.input_size, init_strategy="constant")
        if self.metric_type == 'attention':
            for module in self.linear_sims:
                module.reset_parameters()
        elif self.metric_type == 'weighted_cosine':
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        elif self.metric_type == 'weighted_cosine2':
            self.weight_tensor1 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor1))
            self.weight_tensor2 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor2))
        elif self.metric_type == 'gat_attention':
            for module in self.linear_sims1:
                module.reset_parameters()
            for module in self.linear_sims2:
                module.reset_parameters()
        elif self.metric_type == 'kernel':
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.init.xavier_uniform_(self.weight)
        elif self.metric_type == 'transformer':
            self.linear_sim1.reset_parameters()
            self.linear_sim2.reset_parameters()
        elif self.metric_type == 'cosine':
            pass
        elif self.metric_type == 'mlp':
            self.lin1.reset_parameters()
            self.lin2.reset_parameters()
        elif self.metric_type == 'multi_mlp':
            for module in self.linear_sims1:
                module.reset_parameters()
            for module in self.linear_sims2:
                module.reset_parameters()
        else:
            raise ValueError('Unknown metric_type: {}'.format(self.metric_type))

    def forward(self, node_features):
        if self.feature_denoise:
            masked_features = self.mask_feature(node_features)
            probability, learned_adj = self.learn_adj(masked_features)
            return probability, learned_adj
        else:
            probability, learned_adj = self.learn_adj(node_features)
            return probability, learned_adj

    def learn_adj(self, context, ctx_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)
        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """

        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))
            attention /= len(self.linear_sims)
            markoff_value = -INF

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1) # [num_pers, 1, dim]
            # print('weighted_cosine', self.weight_tensor.grad)
            if len(context.shape) == 3:  # context 指的是节点特征矩阵
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
            # context_fc [num_pers, num_node, dim]
            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1) # 正则
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)  # 先dot求相似度然后，取第一维的平均，就是多头的平均
            markoff_value = 0
            # markoff_value = -INF
        elif self.metric_type == 'weighted_cosine2':
            expand_weight_tensor1 = self.weight_tensor1.unsqueeze(1) # [num_pers, 1, dim]
            expand_weight_tensor2 = self.weight_tensor2.unsqueeze(1)
            if len(context.shape) == 3:  # context 指的是节点特征矩阵
                expand_weight_tensor1 = expand_weight_tensor1.unsqueeze(1)
                expand_weight_tensor2 = expand_weight_tensor2.unsqueeze(1)
            # context_fc [num_pers, num_node, dim]
            context_fc1 = context.unsqueeze(0) * expand_weight_tensor1
            context_fc2 = context.unsqueeze(0) * expand_weight_tensor2
            context_norm1 = F.normalize(context_fc1, p=2, dim=-1)  # 正则
            context_norm2 = F.normalize(context_fc2, p=2, dim=-1)
            attention = torch.matmul(context_norm1, context_norm2.transpose(-1, -2)).mean(0)  # 先dot求相似度然后，取第一维的平均，就是多头的平均
            markoff_value = 0

        elif self.metric_type == 'transformer':
            # print(self.linear_sim1.weight)
            # print('self.linear_sim1.weight', self.linear_sim1.weight.grad)
            Q = self.linear_sim1(context)
            attention = torch.matmul(Q, Q.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
            markoff_value = -INF

        elif self.metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                # print('self.lin1.weight', self.linear_sims2[_].weight.grad)
                a_input1 = self.linear_sims1[_](context)
                a_input2 = self.linear_sims2[_](context)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF
            # markoff_value = 0

        elif self.metric_type == 'kernel':
            # print('self.weight', self.weight.grad)
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self.compute_distance_mat(context, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis**2))
            markoff_value = 0

        elif self.metric_type == 'cosine':
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
            attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).detach()
            markoff_value = 0
        elif self.metric_type == 'mlp':
            # 特征矩阵先做一个MLP，然后 做一个矩阵乘积 dot
            # 输出就算是一个adj
            # print('self.lin1.weight', self.lin1.weight.grad)
            context_fc = torch.relu(self.lin2(torch.relu(self.lin1(context))))
            attention = torch.matmul(context_fc, context_fc.transpose(-1, -2))
            markoff_value = 0
        elif self.metric_type == 'multi_mlp':
            attention = 0
            for _ in range(self.num_pers):
                print('self.lin1.weight', self.linear_sims2[_].weight.grad)
                context_fc = torch.relu(self.linear_sims2[_](torch.relu(self.linear_sims1[_](context))))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))

            attention /= self.num_pers
            markoff_value = -INF
        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        # 计算出节点相似度后，有三种构成图的方式，阈值过滤，KNN和概率采样
        if self.graph_type == 'epsilonNN':
            assert self.epsilon is not None
            out_adj = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)
        elif self.graph_type == 'KNN':
            assert self.top_k is not None
            out_adj = self.build_knn_neighbourhood(attention, self.top_k, markoff_value)
            min_a = torch.min(attention)
            max_a = torch.max(attention)
            attention = (attention - min_a) / (max_a - min_a)
        elif self.graph_type == 'prob':
            out_adj = self.build_prob_neighbourhood(attention, temperature=0.05)
        elif self.graph_type == 'sample1':
            out_adj = attention
        else:
            raise ValueError('Unknown graph_type: {}'.format(self.graph_type))
        if self.graph_type in ['KNN', 'epsilonNN']:
            if self.metric_type in ('kernel', 'weighted_cosine', 'weighted_cosine2'):# , 'weighted_cosine'
                if out_adj.min().item()<0:
                    out_adj = torch.where(out_adj<0, torch.zeros_like(out_adj), out_adj)
                assert out_adj.min().item() >= 0
                #   clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
                out_adj = out_adj / torch.clamp(torch.sum(out_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            elif self.metric_type == 'cosine':
                ouout_adj = (out_adj > 0).float()
                out_adj = normalize_adj(out_adj)
            elif self.metric_type in ('transformer', 'attention', 'gat_attention'):
                out_adj = torch.softmax(out_adj, dim=-1)
        return attention, out_adj

    def build_knn_neighbourhood(self, attention, top_k, markoff_value):
        top_k = min(top_k, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, top_k, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)
        weighted_adjacency_matrix = weighted_adjacency_matrix.to(self.device)
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        attention = torch.sigmoid(attention)
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def build_prob_neighbourhood(self, attention, temperature=0.1):
        # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
        attention = torch.clamp(attention, 0.01, 0.99)
        # 利用 宽松的伯努利分布，设置温度系数和权重 获得最终的抽样概率
        weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).to(attention.device),
                                                     probs=attention).rsample()
        # 如果大量的节点相似度过高，则矩阵会过于稠密
        eps = 0.9#0.5,
        # 下面是抽样概率小于0.5的被过滤掉
        mask = (weighted_adjacency_matrix > eps).detach().float()
        weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def mask_feature(self, x, use_sigmoid=True, marginalize=True):
        feat_mask = (torch.sigmoid(self.feat_mask) if use_sigmoid else self.feat_mask).to(self.device)
        if marginalize:
            std_tensor = torch.ones_like(x, dtype=torch.float) / 2
            mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
            z = torch.normal(mean=mean_tensor, std=std_tensor).to(self.device)
            x = x + z * (1 - feat_mask)
        else:
            x = x * feat_mask
        return x

    def sample_adj(self, adj_logits):
        """ 纯用预测概率伯努利采样出一个adj"""
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling
        # adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature,
        #                                                                  probs=edge_probs).rsample()
        adj_sampled = RelaxedBernoulli(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_bernoulli(self, adj_logits, adj_orig, alpha):
        """ 利用预测概率和原始adj结合，伯努利采样"""
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha * edge_probs + (1 - alpha) * adj_orig
        # sampling
        # adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature,
        #                                                                  probs=edge_probs).rsample()
        adj_sampled = RelaxedBernoulli(temperature=2, probs=edge_probs).rsample()
        #过滤一些低的噪声
        mask = (adj_sampled > 0.2).detach().float()
        adj_sampled = adj_sampled * mask + 0.0 * (1 - mask)
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1) # torch.triu 取出上三角矩阵，这里是为了形成对称矩阵, 貌似不包含对角矩阵
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_round(self, adj_logits, adj_orig, alpha):
        """ 利用预测概率和原始adj结合，Round过滤"""
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha * edge_probs + (1 - alpha) * adj_orig
        # sampling
        adj_sampled = RoundNoGradient.apply(edge_probs)
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_random(self, adj_logits):
        adj_rand = torch.rand(adj_logits.size())
        adj_rand = adj_rand.triu(1)
        adj_rand = torch.round(adj_rand)
        adj_rand = adj_rand + adj_rand.T
        return adj_rand

    def sample_adj_edge(self, adj_logits, adj_orig, change_frac):
        '''利用计算的概率，删除一部分原有的边，加入一部分潜在的边'''
        adj = adj_orig.to_dense() if adj_orig.is_sparse else adj_orig
        n_edges = adj.nonzero().size(0)
        n_change = change_frac#int(n_edges * change_frac / 2)
        # take only the upper triangle
        edge_probs = adj_logits.triu(1)
        edge_probs = edge_probs - torch.min(edge_probs)  # 归一化
        edge_probs = edge_probs / torch.max(edge_probs)
        adj_inverse = 1 - adj  # 原始邻接矩阵的反转
        # get edges to be removed
        mask_rm = edge_probs * adj  # 概率矩阵乘adj，也就是只取出原有的边
        nz_mask_rm = mask_rm[mask_rm > 0] # 只保留概率大于0的边
        # if len(nz_mask_rm) > 0:
        #     n_rm = len(nz_mask_rm) if len(nz_mask_rm) < n_change else n_change
        #     thresh_rm = torch.topk(mask_rm[mask_rm > 0], n_rm, largest=False)[0][-1] # 取出topk的那个阈值
        #     mask_rm[mask_rm > thresh_rm] = 0
        #     mask_rm = CeilNoGradient.apply(mask_rm)
        #     mask_rm = mask_rm + mask_rm.T
        # remove edges
        adj_new = adj# - mask_rm
        # get edges to be added
        mask_add = edge_probs * adj_inverse
        nz_mask_add = mask_add[mask_add > 0]
        if len(nz_mask_add) > 0:
            n_add = len(nz_mask_add) if len(nz_mask_add) < n_change else n_change
            thresh_add = torch.topk(mask_add[mask_add > 0], n_add, largest=True)[0][-1]
            mask_add[mask_add < thresh_add] = 0
            mask_add = CeilNoGradient.apply(mask_add)
            mask_add = mask_add + mask_add.T
        # add edges
        adj_new = adj_new + mask_add
        return adj_new

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()

    @staticmethod
    def backward(ctx, g):
        return g

















