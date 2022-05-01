#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xuhong Wang
# @Email  : wxuhong@amazon.com or wang_xuhong@sjtu.edu.cn
# Feel free to send me an email if you have any question.
# You can also CC Quan Gan (quagan@amazon.com).
import argparse
import logging
import sys
import time

import dgl
import dgl.nn.pytorch as dglnn
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from public_mlps import RelationalMLP_Single
from public_time_encoder import RawDigitsEncoder
from public_utils import load_public_dataset, load_statistics
from public_path_manager import OutPutPathManager
from sklearn.metrics import roc_auc_score
from torch import nn


def get_args():
    # Argument and global variables
    parser = argparse.ArgumentParser('Base')
    parser.add_argument('-exp', '--exp_name', type=str, default="_Test", help='Which expriment does this job belong to')
    parser.add_argument('-job', '--job_name', type=str, default=str(time.time()), help='job name')
    parser.add_argument('--dim_node', type=int, default=128, help='Dimensions of the node embedding and node features')
    parser.add_argument('--dim_time', type=int, default=128, help='Dimensions of the time embedding')
    parser.add_argument('--time_encoder_type', type=str, default="period",
                        choices=["period", "period_E", "digits", "digits_E"], help='Time Encoder')
    parser.add_argument('--embedding', type=str, default="RGCN_SAGE",
                        choices=["RGCN_SAGE"], help='Embedding type')
    "------------------------------------------------------------------------------------------------"
    parser.add_argument('--data_name', type=str, choices=["A", "B"], default='A', help='Dataset name')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument("--n_layers", type=int, default=2, help="number of hidden gnn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args


def get_logger():
    logger = logging.Logger("main")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s \n')
    # ch
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class HeteroConv(nn.Module):
    def __init__(self, etypes, n_layers, in_feats, hid_feats, activation, dropout=0.2):
        super(HeteroConv, self).__init__()
        self.etypes = etypes
        self.n_layers = n_layers
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.act = activation

        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.hconv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers+1):
            self.norms.append(nn.BatchNorm1d(hid_feats))

        # input layer
        self.hconv_layers.append(self.build_hconv(in_feats, hid_feats, activation=self.act))
        # hidden layers
        for i in range(n_layers - 1):
            self.hconv_layers.append(self.build_hconv(hid_feats, hid_feats, activation=self.act))
        # output layer
        self.hconv_layers.append(self.build_hconv(hid_feats, hid_feats))  # activation None

        self.time_encoder = RawDigitsEncoder()

        self.classifier = RelationalMLP_Single(hid_feats*2+10, len(self.etypes))

    def build_hconv(self, in_feats, out_feats, activation=None):
        GNN_dict = {}
        for event_type in self.etypes:
            GNN_dict[event_type] = dglnn.SAGEConv(in_feats=in_feats, out_feats=out_feats,
                                                  aggregator_type='mean', activation=activation)
        return dglnn.HeteroGraphConv(GNN_dict, aggregate='sum')

    def forward(self, g, feat_key='feat'):
        h = g.ndata[feat_key]
        if not isinstance(h, dict):
            h = {'Node': g.ndata[feat_key]}
        for i, layer in enumerate(self.hconv_layers):
            h = layer(g, h)
            for key in h.keys():
                h[key] = self.norms[i](h[key])
        return h

    def emb_concat(self, g, etype):
        def cat(edges):
            return {'emb_cat': torch.cat([edges.src['emb'], edges.dst['emb']], 1)}
        with g.local_scope():
            g.apply_edges(cat, etype=etype)
            emb_cat = g.edges[etype].data['emb_cat']
        return emb_cat

    def time_encoding(self, x):
        '''
        This function is designed to encode a unix timestamp to a 10-dim vector. 
        And it is only one of the many options to encode timestamps.
        Users can also define other time encoding methods such as Neural Network based ones.
        '''
        x = x.unsqueeze(dim=1)
        x = self.time_encoder(x)
        return x.squeeze()

    def time_predict(self, node_emb_cat, time_emb, relation_idx):
        h = torch.cat([node_emb_cat, time_emb], 1)
        return self.classifier(h, relation_idx)


def train(args):

    # init path
    outputPathManager = OutPutPathManager(args.exp_name, args.job_name)
    PATH_CKP_F = outputPathManager.get_checkpoint_path
    PATH_EMB_F = outputPathManager.get_checkpoint_path
    PATH_WANDB = outputPathManager.get_wandb_dir()
    # init wandb
    if args.exp_name == "_Test":
        project = "_Test"
    else:
        project = "test_project"
    wandb.init(project=project, name=args.job_name, config=args, dir=PATH_WANDB)

    # init logger
    logger = get_logger()

    WS_BEST_VALID_AUC = "best_valid_auc"
    WS_BEST_VALID_EPOCH = "best_valid_epoch"
    wandb.summary[WS_BEST_VALID_AUC] = 0
    wandb.summary[WS_BEST_VALID_EPOCH] = 0

    statistics = load_statistics(args.data_name)
    graph = get_dgl_grapph(args.data_name, load_public_dataset(args.data_name, "train", ), statistics, False)
    graph = preprocess(args, graph)
    # these graph are used to train, validate and test classifier
    train_graph = get_dgl_grapph(args.data_name, load_public_dataset(args.data_name, "train", "mixed"), statistics, True)
    valid_graph = get_dgl_grapph(args.data_name, load_public_dataset(args.data_name, "validation", "mixed"), statistics, True)
    test_graph = get_dgl_grapph(args.data_name, load_public_dataset(args.data_name, "test", "mixed"), statistics, True)

    # get model
    model = HeteroConv(graph.etypes, args.n_layers, args.dim_node, args.dim_node, F.relu)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_func = nn.BCEWithLogitsLoss(reduction='sum')

    for epoch_idx in range(args.n_epoch):
        loss = 0
        probs = []
        labels = []
        model.train()
        node_emb = model(graph)

        for ntype in train_graph.ntypes:
            train_graph.nodes[ntype].data['emb'] = node_emb[ntype]

        for i, etype in enumerate(train_graph.etypes):
            emb_cat = model.emb_concat(train_graph, etype)
            ts = train_graph.edges[etype].data['ts']
            time_emb = model.time_encoding(ts)

            prob = model.time_predict(emb_cat, time_emb, int(etype)).squeeze()
            label = train_graph.edges[etype].data['label']

            loss += loss_func(prob, label)

            probs.extend(prob.cpu().detach().numpy().tolist())
            labels.extend(label.cpu().detach().numpy().tolist())

        loss /= train_graph.num_edges()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        train_auc = roc_auc_score(labels, probs)
        wandb.log({"train/loss": loss.item(), "train/auc": train_auc, "epoch": epoch_idx})
        logger.info({"train/loss": loss.item(), "train/auc": train_auc, "epoch": epoch_idx})
        # validate
        valid_loss, valid_auc = evaluate(model, node_emb, valid_graph, loss_func)
        wandb.log({"valid/loss": valid_loss, "valid/auc": valid_auc, "epoch": epoch_idx})
        logger.info({"valid/loss": valid_loss, "valid/auc": valid_auc, "epoch": epoch_idx})
        # save state and node embedding
        if valid_auc > wandb.summary[WS_BEST_VALID_AUC]:
            wandb.summary[WS_BEST_VALID_AUC] = valid_auc
            wandb.summary[WS_BEST_VALID_EPOCH] = epoch_idx
            # save state
            torch.save(model.state_dict(), PATH_CKP_F(epoch_idx))
            dgl.data.utils.save_info(PATH_EMB_F(str(epoch_idx)+'emb'), node_emb)
    # test
    model.load_state_dict(torch.load(PATH_CKP_F(wandb.summary[WS_BEST_VALID_EPOCH])))
    dgl.data.utils.load_info(PATH_EMB_F(str(wandb.summary[WS_BEST_VALID_EPOCH])+'emb'))
    test_loss, test_auc = evaluate(model, node_emb, test_graph, loss_func)
    wandb.summary['test_auc'] = test_auc
    wandb.summary['test_loss'] = test_loss
    logger.info({"test_auc": test_auc, "test_loss": test_loss})


@torch.no_grad()
def evaluate(model, node_emb, eval_graph, loss_fcn):
    with eval_graph.local_scope():
        loss = 0
        probs = []
        labels = []
        for ntype in eval_graph.ntypes:
            eval_graph.nodes[ntype].data['emb'] = node_emb[ntype]

        for i, etype in enumerate(eval_graph.etypes):
            emb_cat = model.emb_concat(eval_graph, etype)
            ts = eval_graph.edges[etype].data['ts']
            time_emb = model.time_encoding(ts)

            prob = model.time_predict(emb_cat, time_emb, int(etype)).squeeze()
            label = eval_graph.edges[etype].data['label']

            loss += loss_fcn(prob, label)
            probs.extend(prob.cpu().numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())
        loss /= eval_graph.num_edges()
        loss = loss.item()
    eval_auc = roc_auc_score(labels, probs)
    return loss, eval_auc


def preprocess(args, directed_g):
    # this function is used to add reverse edges for model computing
    if args.data_name == 'A':
        g = dgl.add_reverse_edges(directed_g, copy_edata=True)
    if args.data_name == 'B':
        graph_dict = {}
        for (src_type, event_type, dst_type) in directed_g.canonical_etypes:
            graph_dict[(src_type, event_type, dst_type)] = directed_g.edges(etype=(src_type, event_type, dst_type))
            src_nodes_reversed = directed_g.edges(etype=(src_type, event_type, dst_type))[1]
            dst_nodes_reversed = directed_g.edges(etype=(src_type, event_type, dst_type))[0]
            graph_dict[(dst_type, event_type+'_reversed', src_type)] = (src_nodes_reversed, dst_nodes_reversed)
        g = dgl.heterograph(graph_dict)
        for etype in g.etypes:
            g.edges[etype].data['ts'] = directed_g.edges[etype.split('_')[0]].data['ts']
            if 'feat' in directed_g.edges[etype.split('_')[0]].data.keys():
                g.edges[etype].data['feat'] = directed_g.edges[etype.split('_')[0]].data['feat']
    # assign node featurs
    for ntype in g.ntypes:
        g.nodes[ntype].data['feat'] = torch.randn((g.number_of_nodes(ntype), args.dim_node)) * 0.05
    return g


def get_dgl_grapph(data_name, edge_csv, statistics, is_evaluation):
    assert data_name in ["A", "B"]
    num_nodes_dict = {}
    if data_name == 'A':
        src_type = 'Node'
        dst_type = 'Node'
        num_nodes_dict['Node'] = statistics['max_src']
    else:
        src_type = 'User'
        dst_type = 'Item'
        num_nodes_dict['User'] = statistics['max_src']
        num_nodes_dict['Item'] = statistics['max_dst']-statistics['max_src']
    # convert node idx to [0,n_src|n_dst)
    edge_csv.iloc[:, 0] -= (statistics["min_src"])
    edge_csv.iloc[:, 1] -= (statistics["min_dst"])
    # for relation idx, convert it to [0,n_relation)
    edge_csv.iloc[:, 2] -= 1

    heterogenous_group = edge_csv.groupby(2)
    graph_dict = {}
    ts_dict = {}
    label_dict = {}
    for event_type, records in heterogenous_group:
        event_type = str(event_type)
        graph_dict[(src_type, event_type, dst_type)] = (records[0].to_numpy(), records[1].to_numpy())
        ts_dict[(src_type, event_type, dst_type)] = (torch.IntTensor(records[3].to_numpy(dtype=np.int32)))
        if is_evaluation:
            label_dict[(src_type, event_type, dst_type)] = (torch.FloatTensor(records[4].to_numpy()))
    g = dgl.heterograph(graph_dict, num_nodes_dict=num_nodes_dict)

    g.edata['ts'] = ts_dict
    g.edata['label'] = label_dict
    return g


if __name__ == "__main__":
    args = get_args()
    train(args)
