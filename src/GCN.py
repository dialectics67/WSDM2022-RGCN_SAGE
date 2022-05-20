# IN DGL, "retain_graph" is useless, FUCK
import argparse
import logging
import sys
import time

import dgl
import dgl.nn.pytorch as dglnn
import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from public_mlps import RelationalMLP
from public_time_encoder import get_time_encoder
from public_utils import load_public_dataset, load_statistics, get_missing_edges_by_ts
from public_path_manager import OutPutPathManager
from sklearn.metrics import roc_auc_score
from torch import nn


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 out_dim,

                 activation,
                 dropout,
                 time_encoder_type,
                 n_time,
                 n_relation
                 ):
        super(GraphSAGE, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # input layer
        self.layers.append(dglnn.GraphConv(in_dim, num_hidden, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(dglnn.GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(dglnn.GraphConv(num_hidden, out_dim))  # activation None

        # downstream models
        self.time_encoder = get_time_encoder(time_encoder_type, n_time)
        self.classifier = RelationalMLP(out_dim*2+n_time, n_relation)

    def forward(self, h):
        for l, layer in enumerate(self.layers):
            h = layer(self.g, h)
            if l != len(self.layers) - 1:
                h = self.dropout(h)
        return h

    def predict(self, node_emb_cat, times, relation_idx):
        times = times.unsqueeze(dim=1)
        time_emb = self.time_encoder(times).squeeze()
        h = torch.cat([node_emb_cat, time_emb], 1)
        return self.classifier(h, relation_idx)


def get_logger():
    logger = logging.Logger("main")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s \n')
    # ch
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def train(args):
    # init path
    outputPathManager = OutPutPathManager(args.exp_name, args.job_name)
    PATH_CKP_F = outputPathManager.get_checkpoint_path
    PATH_WANDB = outputPathManager.get_wandb_dir()
    # init wandb
    if args.exp_name == "_Test":
        project = "_Test"
    else:
        project = "End2End"
    wandb.init(project=project, name=args.job_name, config=args, dir=PATH_WANDB)

    # init logger
    logger = get_logger()

    WS_BEST_VALID_AUC = "best_valid_auc"
    WS_BEST_VALID_EPOCH = "best_valid_epoch"
    wandb.summary[WS_BEST_VALID_AUC] = 0
    wandb.summary[WS_BEST_VALID_EPOCH] = 0

    statistics = load_statistics(args.data_name)
    graph = get_dgl_grapph(args.data_name, load_public_dataset(args.data_name, "train"),
                           statistics, False,
                           drop_duplicates=True).to(args.device)
    graph = preprocess(args, graph)
    # these graph are used to train, validate and test classifier
    train_graph = get_dgl_grapph(args.data_name, load_public_dataset(args.data_name, "train"),
                                 statistics, True,
                                 reconstruct=True, window_size=args.full_bt, unit_ts=statistics['unit_ts']
                                 ).to(args.device)
    valid_graph = get_dgl_grapph(args.data_name, load_public_dataset(
        args.data_name, "validation", "mixed"), statistics, True).to(args.device)
    test_graph = get_dgl_grapph(args.data_name, load_public_dataset(
        args.data_name, "test", "mixed"), statistics, True).to(args.device)

    # get model
    model = GraphSAGE(graph, args.num_layers, args.dim_node, args.dim_node, args.dim_node,
                      F.relu, args.dropout, args.time_encoder_type,
                      args.dim_time, n_relation=statistics["max_rel"]
                      ).to(args.device)
    parameters = [{'params': model.parameters()}]
    if args.learn_node_feats:
        parameters.append({'params': graph.ndata['feat']})
    optimizer = torch.optim.AdamW(parameters,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    # resume point
    loss_func = nn.BCEWithLogitsLoss(reduction='sum')

    # iter all epoch
    n_full_batch = 0
    for epoch_idx in range(args.n_epoch):
        # iter all full_batch
        model.train()
        node_emb = model(graph.ndata['feat'])  # remeber to recalcute node_emb after every
        for full_st_ts in range(train_graph.edata['ts'].min(), train_graph.edata['ts'].max()+1, args.full_bt):
            full_ed_ts = min(full_st_ts+args.full_bt, train_graph.edata['ts'].max()+1)
            full_edge_mask = torch.logical_and(train_graph.edata['ts'] >= full_st_ts, train_graph.edata['ts'] < full_ed_ts)
            full_edge_idx = torch.arange(0, full_edge_mask.shape[0], dtype=torch.int64)[full_edge_mask]
            full_loss_sum = 0
            full_labels = train_graph.edata['label'][full_edge_idx].cpu().numpy()
            full_probs = []
            for mini_st_idx in range(0, len(full_edge_idx), args.mini_bs):
                mini_ed_idx = min(mini_st_idx+args.mini_bs, len(full_edge_idx))
                mini_edge_idx = full_edge_idx[mini_st_idx:mini_ed_idx]

                src_emb = node_emb[train_graph.edges()[0][mini_edge_idx]]
                dst_emb = node_emb[train_graph.edges()[1][mini_edge_idx]]
                relation = train_graph.edata['relation'][mini_edge_idx]
                ts = train_graph.edata['ts'][mini_edge_idx]
                label = train_graph.edata['label'][mini_edge_idx]

                prob_scores = model.predict(torch.cat([src_emb, dst_emb], dim=1), ts, relation)
                mini_loss_sum = loss_func(prob_scores, label)
                mini_loss_mean = mini_loss_sum/len(label)
                if args.opt_method == "mini_batch":
                    mini_loss_mean.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    node_emb = model(graph.ndata['feat'])
                else:
                    (mini_loss_sum/train_graph.num_edges()).backward()
                    node_emb = model(graph.ndata['feat'])

                full_loss_sum += mini_loss_sum.item()
                full_probs.extend(prob_scores.sigmoid().cpu().detach().numpy().tolist())

            if args.opt_method == 'full_batch':
                optimizer.step()
                optimizer.zero_grad()
                node_emb = model(graph.ndata['feat'])
            full_auc = roc_auc_score(full_labels, full_probs)
            wandb.log({"full_batch/loss": full_loss_sum/len(full_labels),
                      "full_batch/auc": full_auc, "full_batch": n_full_batch})
            logger.info({"full_batch/loss": full_loss_sum/len(full_labels),
                        "full_batch/auc": full_auc, "full_batch": n_full_batch})
            n_full_batch += 1

        # validate
        valid_loss, valid_auc = evaluate(model, graph, valid_graph, loss_func, args.mini_bs)
        wandb.log({"valid/loss": valid_loss, "valid/auc": valid_auc, "epoch": epoch_idx})
        logger.info({"valid/loss": valid_loss, "valid/auc": valid_auc, "epoch": epoch_idx})
        # save state and node embedding
        if valid_auc > wandb.summary[WS_BEST_VALID_AUC]:
            wandb.summary[WS_BEST_VALID_AUC] = valid_auc
            wandb.summary[WS_BEST_VALID_EPOCH] = epoch_idx
            # save state
            checkpoint = {
                "model": model.state_dict(),
                "node_feat": graph.ndata['feat']
            }
            torch.save(checkpoint, PATH_CKP_F(epoch_idx))

    # test
    checkpoint = torch.load(PATH_CKP_F(wandb.summary[WS_BEST_VALID_EPOCH]))
    model.load_state_dict(checkpoint['model'])
    graph.ndata['feat'] = checkpoint['node_feat']
    test_loss, test_auc = evaluate(model, graph, test_graph, loss_func, args.mini_bs)
    wandb.summary['test_loss'] = test_loss
    wandb.summary['test_auc'] = test_auc
    logger.info({"test_loss": test_loss, "test_auc": test_auc})


@torch.no_grad()
def evaluate(model, graph, eval_graph, loss_fcn, bs):
    model.eval()
    node_emb = model(graph.ndata['feat'])
    with eval_graph.local_scope():
        loss = 0
        probs = []
        labels = []
        for st_idx in range(0, eval_graph.num_edges(), bs):
            ed_idx = min(st_idx+bs, eval_graph.num_edges())
            src_emb = node_emb[eval_graph.edges()[0][st_idx:ed_idx]]
            dst_emb = node_emb[eval_graph.edges()[1][st_idx:ed_idx]]
            relation = eval_graph.edata['relation'][st_idx:ed_idx]
            ts = eval_graph.edata['ts'][st_idx:ed_idx]
            label = eval_graph.edata['label'][st_idx:ed_idx]

            prob_scores = model.predict(torch.cat([src_emb, dst_emb], dim=1), ts, relation)

            loss += loss_fcn(prob_scores, label)
            labels.extend(label.cpu().numpy().tolist())
            probs.extend(prob_scores.cpu().numpy().tolist())
        loss /= eval_graph.num_edges()
        loss = loss.item()
    eval_auc = roc_auc_score(labels, probs)
    return loss, eval_auc


def preprocess(args, directed_g):
    # this function is used to add reverse edges for model computing
    g = dgl.add_reverse_edges(directed_g, copy_edata=True)
    # assign node featurs
    g.ndata['feat'] = torch.randn((g.number_of_nodes(), args.dim_node), device=g.device)
    if args.learn_node_feats:
        g.ndata['feat'].requires_grad_(True)
    return g


def get_dgl_grapph(data_name, edge_csv: pd.DataFrame, statistics, is_evaluation, *,
                   reconstruct=False, window_size=None, unit_ts=None,
                   drop_duplicates=False):
    assert data_name in ["A", "B"]
    if drop_duplicates:
        edge_csv.drop_duplicates(subset=[0, 1, 2], inplace=True, ignore_index=True)
    if reconstruct:
        edge_csv.sort_values(by=[3], inplace=True)
        edge_csv_tmp = []
        for st_ts in range(edge_csv[3].min(), edge_csv[3].max()+1, window_size):
            ed_ts = min(st_ts+window_size, edge_csv[3].max()+1)
            ts_mask = (edge_csv[3] >= st_ts) & (edge_csv[3] < ed_ts)
            positive_edges = edge_csv.loc[ts_mask]
            negative_edges = get_missing_edges_by_ts(positive_edges.to_numpy(), np.arange(st_ts, ed_ts, unit_ts, dtype=np.int32))
            negative_edges = pd.DataFrame(negative_edges)
            if negative_edges.shape[0] == 0:
                negative_edges = pd.DataFrame(np.array((1, 1, 1, positive_edges[3].max()), dtype=np.int32)).T
            positive_edges.loc[:, 4] = np.array(1, dtype=np.int32)
            negative_edges.loc[:, 4] = np.array(0, dtype=np.int32)
            edge_csv_tmp.append(positive_edges)
            edge_csv_tmp.append(negative_edges)
        edge_csv = pd.concat(edge_csv_tmp, ignore_index=True)
        edge_csv = edge_csv.iloc[np.random.permutation(np.arange(len(edge_csv)))]
        edge_csv.reset_index(inplace=True, drop=True)
    # convert node idx to [0,n_src|n_dst)
    edge_csv.iloc[:, 0] -= (statistics["min_src"])
    edge_csv.iloc[:, 1] -= (statistics["min_src"])
    # for relation idx, convert it to [0,n_relation)
    edge_csv.iloc[:, 2] -= 1

    graph_dict = (edge_csv[0].to_numpy(), edge_csv[1].to_numpy())
    relaion_dict = (torch.IntTensor(edge_csv[2].to_numpy(dtype=np.int32)))
    ts_dict = (torch.IntTensor(edge_csv[3].to_numpy(dtype=np.int32)))
    if is_evaluation:
        label_dict = (torch.FloatTensor(edge_csv[4].to_numpy(dtype=np.int32)))

    g = dgl.graph(graph_dict)
    g.edata['relation'] = relaion_dict
    g.edata['ts'] = ts_dict
    if is_evaluation:
        g.edata['label'] = label_dict
    return g


def get_args():
    # Argument and global variables
    def str2bool(str):
        return True if str.lower() == 'true' else False
    "------------------------------------------------------------------------------------------------"
    "parameters about job"
    parser = argparse.ArgumentParser('Base')
    parser.add_argument('-exp', '--exp_name', type=str, default="_Test", help='Which expriment does this job belong to')
    parser.add_argument('-job', '--job_name', type=str, default=str(time.time()), help='job name')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU')
    parser.add_argument('--data_name', type=str, choices=["A", "B"], default='A', help='Dataset name')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')

    "------------------------------------------------------------------------------------------------"
    "parameters about training strategy"
    parser.add_argument('--full_bt', type=int, default=7*24*3600, help='Time Length of a Full Batch')
    parser.add_argument('--mini_bs', type=int, default=10240, help='Batch Size of a Mini Batch')
    parser.add_argument('--opt_method', type=str, default="full_batch", choices=['mini_batch', 'full_batch'],
                        help='How to optimize the parameters')
    parser.add_argument('--learn_node_feats', type=str2bool, default=True, help='Whether to learn node feats')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--drop_duplicates", type=str2bool, default=True,
                        help='wether drop duplicate neighbors by (src, dst, relation)')

    "------------------------------------------------------------------------------------------------"
    "parameters about model"
    parser.add_argument('--model', type=str, default="SAGE", choices=["SAGE"], help='Embedding type')
    parser.add_argument('--dim_node', type=int, default=128, help='Dimensions of the node embedding and node features')
    parser.add_argument('--dim_time', type=int, default=128, help='Dimensions of the time embedding')
    parser.add_argument('--time_encoder_type', type=str, default="digits_E",
                        choices=["period", "period_E", "digits", "digits_E"], help='Time Encoder')
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")

    
    parser.add_argument("--dropout", type=float, default=0.2, help="model droop")

    try:
        args = parser.parse_args()
        device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
        args.device = torch.device(device_string)
    except:
        parser.print_help()
        sys.exit(0)
    return args


if __name__ == "__main__":
    args = get_args()
    train(args)
