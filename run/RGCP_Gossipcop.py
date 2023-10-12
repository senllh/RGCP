import argparse
import time
import numpy as np
from tqdm import tqdm
import copy as cp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import DataLoader, DataListLoader
from scipy.sparse import coo_matrix
from utils.data_loader import *
from GCL.models import SingleBranchContrast, DualBranchContrast, BootstrapContrast
import GCL.augmentors as A
import GCL.losses as L
from Evaluator import Logistic_classify, svc_classify, \
    randomforest_classify, linearsvc_classify, MLP_classify, get_metrics
from graph_augs import mask_edge, encoding_mask_noise, Learned_mask_edge, Learned_drop_node
from graph_learner import GraphGenerator, ViewLearner
from Graph_structure_learn_layers import GraphLearner
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import random_split, Subset
from utils.eval_helper import eval_deep, acc_f1, metrics, few_shot_split
import math

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        if args.gnn == 'gcn':
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(GCNConv(input_dim, hidden_dim))
                else:
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))

        if args.gnn == 'gat':
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(GATConv(input_dim, hidden_dim))
                else:
                    self.layers.append(GATConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            zs.append(z)
        z = zs[-1]
        return z


class Graph_Refine(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(Graph_Refine, self).__init__()
        self.GraphLearner = GraphLearner(input_size=input_dim, hidden_size=hid_dim,
                                graph_type=args.graph_type, top_k=args.top_k, epsilon=args.epsilon,
                                num_pers=args.num_per, metric_type=args.graph_metric_type,
                                feature_denoise=args.feature_denoise, device=args.device)

    def purified(self, sim, ori_g, threshold=0):
        sim = sim * ori_g
        new_sim = torch.where(sim > threshold, ori_g, torch.zeros_like(sim))
        return new_sim

    def forward(self, x, ori_index, ori_weight):
        node_num = x.size(0)
        ori_g = to_dense_adj(ori_index, edge_attr=ori_weight, max_num_nodes=node_num).squeeze(0)
        sim, sim_g = self.GraphLearner(x)
        if args.feature_denoise == True and args.epsilon > 0.0:
            pur_g = self.purified(sim, ori_g, args.epsilon)
            new_g = pur_g + sim_g
        else:
            new_g = ori_g + sim_g

        new_index, new_weight = dense_to_sparse(new_g)
        return new_index, new_weight

class Contrast(torch.nn.Module):
    def __init__(self, args):
        super(Contrast, self).__init__()
        self.args = args
        self.text_features = args.num_features
        self.user_features = 10
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.gnn = args.gnn
        self.layers = args.layers
        self.bn0 = torch.nn.BatchNorm1d(self.text_features)
        self.bn1 = torch.nn.BatchNorm1d(self.text_features)
        self.encoder = GConv(self.nhid, self.nhid, num_layers=args.layers)
        # project layer
        self.local_mlp = FC(input_dim=self.nhid, output_dim=self.nhid)
        self.global_mlp = FC(input_dim=self.nhid, output_dim=self.nhid)
        # pre-training loss function
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(0.2), mode='G2L').to(args.device)
    def forward(self, x, x1, edge, edge1, batch, encoder=None, weight=None, weight1=None):
        # GNN encoding
        if encoder != None:
            self.encoder = encoder
        z = self.encoder(x, edge, weight)
        z1 = self.encoder(x1, edge1, weight1)

        # Readout
        g = gmp(z, batch)
        g1 = gmp(z1, batch)

        # project
        pro_z, pro_z1 = [self.local_mlp(h) for h in [z, z1]]
        pro_g, pro_g1 = [self.global_mlp(g) for g in [g, g1]]

        # loss
        loss = self.contrast_model(h1=pro_z, h2=pro_z1, g1=pro_g, g2=pro_g1, batch=batch)
        return loss, z, g
#
class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.text_features = args.num_features
        self.user_features = 10
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        # encoder
        self.con_encoder = GConv(self.nhid, self.nhid, num_layers=args.layers)

        self.layers = args.layers
        self.bn0 = torch.nn.BatchNorm1d(self.text_features)
        self.Aug_learner = ViewLearner(self.nhid)
        self.contrast = Contrast(args)
        self.aug_ratio = args.aug_ratio
        self.Aug_learner = ViewLearner(self.nhid)
        self.G_refine = Graph_Refine(self.text_features, self.nhid)
        self.lin1 = nn.Linear(self.nhid, self.nhid)
        self.lin2 = nn.Linear(self.nhid, 2)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.text_features, self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, self.nhid))

    def forward(self, data):
        text_feature, ori_index, batch = data.x, data.edge_index, data.batch
        data.x, node_num, ori_weight = text_feature, text_feature.size(0), torch.ones_like(ori_index[0]).float()

        x = self.bn0(text_feature)
        x = self.mlp(x)

        # Propagation Structure refinement
        ori_index, ori_weight = self.G_refine(text_feature, ori_index, ori_weight)

        # Data augumention
        keep_edges, masked_edges, keep_weight, masked_weight = mask_edge(ori_index, self.aug_ratio, ori_weight)
        masked_x, (mask_id, keep_id) = encoding_mask_noise(x, self.aug_ratio)

        # Contrastive learning
        con_loss, z, g = self.contrast(x, masked_x, ori_index, keep_edges, batch, self.con_encoder, ori_weight, keep_weight)
        return con_loss, g


    def get_embeddings(self, data):
        text_feature, ori_index, batch = data.x, data.edge_index, data.batch

        data.x, node_num, ori_weight = text_feature, text_feature.size(0), torch.ones_like(ori_index[0]).float()

        x = self.bn0(text_feature)
        x = self.mlp(x)

        if args.if_GSL == True:
            ori_index, ori_weight = self.G_refine(text_feature, ori_index, ori_weight)

        z = self.con_encoder(x, ori_index, ori_weight)
        g = gmp(z, batch)

        return z, g, (ori_index, ori_weight)


class MLP(nn.Module):
    def __init__(self, infeats, hid_feats):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(infeats, infeats)
        self.lin2 = nn.Linear(infeats, hid_feats)
        self.active = nn.PReLU()
    def forward(self, h):
        h1 = self.active(self.lin1(h))
        h2 = self.active(self.lin2(h1))
        return h2

class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[: -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context

class Learnable_prompt(nn.Module):
    def __init__(self, infeats, hid_feats):
        super(Learnable_prompt, self).__init__()
        self.learnable_type = args.learnable_type
        self.num_head = args.num_head

        self.MLP = MLP(infeats, hid_feats)

        if self.learnable_type == 'Multi_MLP':
            self.Multi_MLP = nn.ModuleList([MLP(infeats, hid_feats) for _ in range(self.num_head)])
        elif self.learnable_type == 'Linear':
            self.linear = nn.Linear(infeats, hid_feats)
        elif self.learnable_type == 'Multi_weight':
            self.weight_tensor = torch.Tensor(self.num_head, infeats)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        elif self.learnable_type == 'attention':
            self.attention = nn.ModuleList([MLP(infeats, hid_feats) for _ in range(self.num_head)])
        elif self.learnable_type == 'selfAttention':
            self.selfAttention = selfAttention(self.num_head, infeats, hid_feats)
        elif self.learnable_type == 'GCN':
            self.gnn = GConv(infeats, hid_feats, num_layers=2)
        self.active = nn.PReLU()

    def forward(self, g_embed, h, batch, edge_index=None, edge_weight=None):

        if self.learnable_type == 'Multi_MLP':
            hs = []
            for i in range(self.num_head):
                hs.append(self.Multi_MLP[i](h))
            h2 = torch.stack(hs, dim=0).mean(0)
        if self.learnable_type == 'MLP':
            h2 = self.MLP(h)
        if self.learnable_type == 'Linear':
            h2 = self.linear(h)
        elif self.learnable_type == 'Multi_weight':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            h2 = (h.unsqueeze(0) * expand_weight_tensor).mean(0)
        elif self.learnable_type == 'attention':
            hs = []
            for i in range(self.num_head):
                hs.append(torch.relu(self.attention[i](h)))
            h2 = torch.stack(hs, dim=0).mean(0)
        elif self.learnable_type == 'selfAttention':
            h2 = self.selfAttention(h.unsqueeze(1)).squeeze(1)
        elif self.learnable_type == 'GCN':
            h2 = self.gnn(h, edge_index, edge_weight)

        if args.Neighbor == True:
            one_hop_graph = torch.sparse_coo_tensor(edge_index, edge_weight, (h.size(0), h.size(0)))
            for hop in range(2):
               h2 = torch.spmm(one_hop_graph, h2)

        g2 = gmp(h2, batch)

        return g2, h2

def class_test(model, Prompt, loader, event_c_embedding=None, post_c_embedding=None):
    model.eval()
    Prompt.eval()
    loss_test = 0.0
    out_log = []
    test_pred = []
    test_y = []
    g_list = []
    with torch.no_grad():
        for data in loader:
            data = data.to(args.device)
            h, g, (edge_index, edge_weight) = model.get_embeddings(data)
            g1, h1 = Prompt(g, h, data.batch, edge_index, edge_weight)

            if args.Neighbor == True:
                one_hop_graph = torch.sparse_coo_tensor(edge_index, edge_weight, (h.size(0), h.size(0)))
                for hop in range(2):
                    h1 = torch.spmm(one_hop_graph, h1)

            if args.prompt_answer_readout == 'max':
                g1 = gmp(h1, data.batch)
                out = sim(g1, event_c_embedding)

            elif args.prompt_answer_readout == 'mean':
                g1 = global_mean_pool(h1, data.batch)
                out = sim(g1, event_c_embedding)

            g_list.append(g1)
            y = data.y
            out_log.append([F.softmax(out, dim=1), y])
            test_pred.append(F.softmax(out, dim=1))
            test_y.append(y)
            loss_test += F.nll_loss(out, y).item()
    test_pred = torch.cat(test_pred)
    test_y = torch.cat(test_y)

    return metrics(test_pred, test_y), loss_test

def sim(x, y):
    norm_x = F.normalize(x, dim=-1)
    norm_y = F.normalize(y, dim=-1)
    return torch.matmul(norm_x, norm_y.transpose(1,0))

def center_embedding(input, index):
    device = input.device
    index = index.unsqueeze(1)
    mean = torch.ones(index.size(0), index.size(1)).to(device)
    label_num = 2
    _mean = torch.zeros(label_num, 1,device=device).scatter_add_(dim=0, index=index, src=mean)
    preventnan = torch.ones(_mean.size(), device=device)*0.0000001
    _mean = _mean + preventnan
    index = index.expand(input.size())
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index, src=input)
    c = c / _mean
    return c

def train_model(dataset):
    best_loss = 1e9
    cnt_wait = 0
    model = Model(args)
    model = model.to(args.device)
    data_loader = loader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_data_loader = loader(dataset, batch_size=32, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.con_lr, weight_decay=args.con_weight_decay)# we_de=0.0 更好

    '''--------------------pre-traing task--------------------'''
    if args.pre_train == True:
        for epoch in tqdm(range(args.epochs)):
            loss_train = 0.0
            for i, data in enumerate(data_loader):
                data = data.to(args.device)
                model.train()
                optimizer.zero_grad()
                con_loss, _ = model(data)
                loss = con_loss
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            if loss_train < best_loss:
                best_loss = loss_train
                best_epoch = epoch
                cnt_wait = 0
                best_model = model.state_dict()
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print('Early stopping! Best epoch:', best_epoch)
                break
        torch.save(best_model, 'save_model' + '/' + args.dataset + '_' + args.gnn + '_' + args.model_name + '.pkl')

    # 无监督
    '''--------------------unsupervised vaild task--------------------'''
    if args.unsupervised_task == True:
        model.load_state_dict(torch.load('save_model' + '/' + args.dataset + '_' + args.gnn + '_' + args.model_name + '.pkl'))
        model.eval()
        x = []
        y = []
        for i, data in enumerate(test_data_loader):
            data = data.to(args.device)
            _, g, _ = model.get_embeddings(data)
            x.append(g)
            y.append(data.y)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()

        # Machine Learning validation
        linearsvc_classify(x, y, K=5, seed=args.seed)
        linearsvc_classify(x, y, K=10, seed=args.seed)
        linearsvc_classify(x, y, K=20, seed=args.seed)
        print("--" * 50)
        svc_classify(x, y, K=5, seed=args.seed)
        svc_classify(x, y, K=10, seed=args.seed)
        svc_classify(x, y, K=20, seed=args.seed)
        print("--" * 50)
        Logistic_classify(x, y, K=5, seed=args.seed)
        Logistic_classify(x, y, K=10, seed=args.seed)
        Logistic_classify(x, y, K=20, seed=args.seed)

    # 微调
    '''--------------------fine tuning task--------------------'''
    if args.cross_validation_task == True:
        kf = StratifiedKFold(n_splits=args.K_fold, shuffle=True, random_state=args.seed)#args.seed
        sample_num = len(dataset)
        ac_list, pre_list, recall_list, f1_list = [], [], [], []
        prompt_loss = DualBranchContrast(loss=L.InfoNCE(0.2), mode='G2L').to(args.device)
        fold = 0
        for train_index, test_index in kf.split(range(sample_num), dataset.data.y):
            fold += 1
            model.load_state_dict(torch.load('save_model' + '/' + args.dataset + '_' + args.gnn + '_' + args.model_name + '.pkl'))
            Prompt = Learnable_prompt(args.nhid, args.nhid).to(args.device) #128

            # 多头MLP
            Prompt_optimizer = torch.optim.Adam(Prompt.parameters(), lr=args.downstream_lr, weight_decay=args.downstream_weight_decay)  # 1e-2
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=Prompt_optimizer, step_size=5, gamma=0.99)

            best_f1 = 0.0
            class_wait = 0
            train_index, test_index = test_index, train_index
            train_set, test_set = Subset(dataset, train_index), Subset(dataset, test_index)
            class_train_loader = loader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
            class_test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
            loss_record = []
            for class_epoch in tqdm(range(50)):
                loss_train = 0.0
                train_pred = []
                train_embed, train_y = [], []
                model.eval()
                # model.train()
                Prompt.train()
                for i, data in enumerate(class_train_loader):
                    data = data.to(args.device)
                    h, g, (edge_index, edge_weight) = model.get_embeddings(data)
                    g1, h1 = Prompt(g, h, data.batch, edge_index, edge_weight)
                    # event class prototypes
                    event_c_embedding = center_embedding(g, data.y)

                    # post class prototypes
                    post_c_embedding = []
                    for i in range(2):
                        same_c_gid = (data.y == i).nonzero().to(args.device)
                        pos_id = torch.where(data.batch == same_c_gid, 1, 0).nonzero()[:, 1]
                        post_c_embedding.append(torch.mean(h[pos_id], dim=0))
                    post_c_embedding = torch.stack(post_c_embedding, dim=0)

                    train_embed.append(g1)
                    train_y.append(data.y)

                    # Loss of prompt tuning
                    p_loss = prompt_loss(h1=h1, h2=post_c_embedding[data.y][data.batch], g1=g1, g2=event_c_embedding[data.y], batch=data.batch)
                    pre_train = sim(g1, event_c_embedding)

                    loss = p_loss
                    Prompt_optimizer.zero_grad()
                    loss.backward()
                    loss_train += loss.item()/(data.batch.max()+1).item()
                    Prompt_optimizer.step()
                    train_pred.append(pre_train)

                    if args.Neighbor == True:
                        one_hop_graph = torch.sparse_coo_tensor(edge_index, edge_weight, (h.size(0), h.size(0)))
                        for hop in range(2):
                            h1 = torch.spmm(one_hop_graph, h1)
                    if args.prompt_answer_readout == 'max':
                        g1 = gmp(h1, data.batch)
                    elif args.prompt_answer_readout == 'mean':
                        g1 = global_mean_pool(h1, data.batch)
                    train_embed = g1

                loss_record.append(loss_train)
                scheduler.step()
                train_pred = torch.cat(train_pred)
                train_y = torch.cat(train_y)
                acc_train, _, _, _, recall_train = metrics(train_pred, train_y)
                if (class_epoch + 1) % 1 == 0:
                    (acc, f1_macro, f1_micro, precision, recall), test_loss = class_test(model, Prompt, class_test_loader, event_c_embedding=event_c_embedding, post_c_embedding=post_c_embedding)
                    if f1_macro > best_f1:
                        best_f1 = f1_macro
                        best_epoch = class_epoch + 1
                        best_ac = acc
                        best_pre = precision
                        best_recall = recall
                        class_wait = 0
                    else:
                        class_wait += 1
                    if class_wait == 10:
                        print('Early stopping!')
                        break

            ac_list.append(best_ac)
            pre_list.append(best_pre)
            recall_list.append(best_recall)
            f1_list.append(best_f1)
            print(f'Test set results: acc: {best_ac:.4f}, prec: {best_pre:.4f}, recall: {best_recall:.4f}, '
                  f'f1_macro: {best_f1:.4f}, best_epoch: {best_epoch:.1f}')
        print(f1_list)
        print(5, 'Kold ', '----Acc:', np.mean(ac_list), '|Recall:', np.mean(recall_list), '|Prec', np.mean(pre_list),
              '|F1_ma', np.mean(f1_list))

    '''--------------------few_shot task--------------------'''
    intance_loss = DualBranchContrast(loss=L.InfoNCE(0.2), mode='G2G').to(args.device)
    if args.few_shot_task == True:
        prompt_loss = DualBranchContrast(loss=L.InfoNCE(0.2), mode='G2L').to(args.device)
        ac_list, pre_list, recall_list, f1_list = [], [], [], []
        for i in range(50):
            model.load_state_dict(
                torch.load('save_model' + '/' + args.dataset + '_' + args.gnn + '_' + args.model_name + '.pkl'))
            Prompt = Learnable_prompt(args.nhid, args.nhid).to(args.device)
            # 多头MLP
            Prompt_optimizer = torch.optim.Adam(Prompt.parameters(), lr=args.downstream_lr,
                                                weight_decay=args.downstream_weight_decay)  # 1e-2
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=Prompt_optimizer, step_size=5, gamma=0.99)

            best_f1 = 0.0
            class_wait = 0
            train_index, test_index = few_shot_split(dataset, args.num_shot, args.num_classes)
            train_set, test_set = Subset(dataset, train_index), Subset(dataset, test_index)
            class_train_loader = loader(train_set, batch_size=64, shuffle=True, num_workers=0)
            class_test_loader = loader(test_set, batch_size=64, shuffle=False, num_workers=0)
            for class_epoch in tqdm(range(50)):
                train_embed, train_y = [], []
                model.eval()
                Prompt.train()
                for i, data in enumerate(class_train_loader):
                    data = data.to(args.device)
                    h, g, (edge_index, edge_weight) = model.get_embeddings(data)
                    out, g1, h1 = Prompt(g, h, data.batch, edge_index, edge_weight)

                    # event class prototypes
                    event_c_embedding = center_embedding(g, data.y)

                    # post class prototypes
                    post_c_embedding = []
                    for i in range(2):
                        same_c_gid = (data.y == i).nonzero().to(args.device)
                        pos_id = torch.where(data.batch == same_c_gid, 1, 0).nonzero()[:, 1]
                        post_c_embedding.append(torch.mean(h[pos_id], dim=0))
                    post_c_embedding = torch.stack(post_c_embedding, dim=0)

                    train_embed.append(g1)
                    train_y.append(data.y)
                    p_loss = prompt_loss(h1=h1, h2=post_c_embedding[data.y][data.batch], g1=g1, g2=event_c_embedding[data.y], batch=data.batch)
                    loss = p_loss
                    Prompt_optimizer.zero_grad()
                    loss.backward()
                    Prompt_optimizer.step()
                scheduler.step()
                if (class_epoch + 1) % 1 == 0:
                    (acc, f1_macro, f1_micro, precision, recall), test_loss = class_test(model, Prompt,
                                                                                         class_test_loader,
                                                                                         c_embedding=event_c_embedding,
                                                                                         node_embedding=post_c_embedding)
                    if f1_macro > best_f1:
                        best_f1 = f1_macro
                        best_epoch = class_epoch + 1
                        best_ac = acc
                        best_pre = precision
                        best_recall = recall
                        class_wait = 0
                    else:
                        class_wait += 1
                    if class_wait == 10:
                        print('Early stopping!')
                        break
            ac_list.append(best_ac)
            pre_list.append(best_pre)
            recall_list.append(best_recall)
            f1_list.append(best_f1)
            print(f'Test set results: acc: {best_ac:.4f}, prec: {best_pre:.4f}, recall: {best_recall:.4f}, '
                  f'f1_macro: {best_f1:.4f}, best_epoch: {best_epoch:.1f}')
        print(f1_list)
        ac_list, pre_list, recall_list, f1_list = np.array(ac_list), np.array(pre_list), np.array(
            recall_list), np.array(f1_list)
        print(
            f'Mean:, acc: {np.mean(ac_list):.4f}, prec: {np.mean(pre_list):.4f}, recall: {np.mean(recall_list):.4f}, '
            f'f1_macro: {np.mean(f1_list):.4f}')
        print(
            f'Standard Deviation:, acc: {np.std(ac_list):.4f}, prec: {np.std(pre_list):.4f}, recall: {np.std(recall_list):.4f}, '
            f'f1_macro: {np.std(f1_list):.4f}')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023, help='random seed') #2023, 123, 777, 888, 999
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
# hyper-parameters
parser.add_argument('--dataset', type=str, default='gossipcop', help='[politifact, gossipcop]')
parser.add_argument('--aug_ratio', type=float, default=0.5, help='argument ratio')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--prompt_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--con_lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--struc_lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--feat_lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--con_weight_decay', type=float, default=0.000, help='weight decay')
parser.add_argument('--struc_weight_decay', type=float, default=0.00, help='weight decay')
parser.add_argument('--feat_weight_decay', type=float, default=0.00, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs, politifact:350')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content, all]')
parser.add_argument('--gnn', type=str, default='gcn', help='model type, [gcn, gat, sage, gin]')
parser.add_argument('--patience', type=int, default=10, help='patience')
parser.add_argument('--layers', type=int, default=2, help='The layers of GNNs')
parser.add_argument('--model_name', type=str, default='RGCP', help='')
# Graph refine
parser.add_argument("--graph_type", type=str, default="KNN", help="epsilonNN, KNN, prob, sample1")
parser.add_argument("--graph_metric_type", type=str, default="weighted_cosine") # weighted_cosine, weighted_cosine2 , mlp, kernel, multi_mlp, transformer, gat_attention
parser.add_argument("--repar", type=bool, default=True, help="Default is True.")
parser.add_argument("--top_k", type=int, default=10, help="Default is 10.")
parser.add_argument("--graph_skip_conn", type=float, default=0.0, help="Default is 0.0.")
parser.add_argument("--num_per", type=int, default=2, help="Default is 16")
parser.add_argument("--epsilon", type=float, default=0.8, help="Default is 0.8.")
parser.add_argument("--feature_denoise", type=bool, default=True, help='True, False' )
# pre-training
parser.add_argument('--pre_train', type=bool, default=True, help='True, False')

# Prompting
parser.add_argument('--downstream_task', type=str, default='prompting', help='prompting, fine_tuning')
parser.add_argument('--learnable_type', type=str, default='MLP', help='Multi_MLP, Multi_weight, attention, selfAttention, GCN')
parser.add_argument('--num_head', type=int, default='4', help='the number of head ')
parser.add_argument('--prompt_answer_readout', type=str, default='max', help='mean, max')
parser.add_argument('--downstream_lr', type=float, default=0.01, help='the number of head ')
parser.add_argument('--downstream_weight_decay', type=float, default=1e-5, help='the number of head ')
parser.add_argument('--Neighbor', type=bool, default=True, help='True, False')

# Downstream task
parser.add_argument("--unsupervised_task", type=bool, default=False, help="Default is False.")
parser.add_argument("--fine_funning_task", type=bool, default=True, help="Default is False.")
parser.add_argument("--few_shot_task", type=bool, default=False, help="Default is False.")
parser.add_argument("--num_shot", type=int, default=5, help="Default is False.")
parser.add_argument("--K_fold", type=int, default=5, help="K fold.")
parser.add_argument('--alpha', type=float, default=0.0, help='')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
dataset = FNNDataset(root='../data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
print(args)

if args.multi_gpu:
    loader = DataListLoader
else:
    loader = DataLoader

if __name__ == '__main__':
    train_model(dataset)
