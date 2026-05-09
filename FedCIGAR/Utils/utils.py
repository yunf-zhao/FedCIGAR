import torch
import torch.nn.functional as F
import numpy as np
import random
import math
from torch_geometric.utils import to_networkx, degree, to_scipy_sparse_matrix,to_dense_adj
from scipy import sparse as sp
import os
import csv
import json

def normalize_features(x):
    row_sum = x.sum(1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    x_norm = x / row_sum
    return x_norm



def init_structure_encoding(args, gs, type_init,is_norm=False):
    if type_init == 'rw':
        for g in gs:
            # Geometric diffusion features with Random Walk
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv = sp.diags(D)
            RW = A * Dinv
            M = RW

            SE_rw = [torch.from_numpy(M.diagonal()).float()]
            M_power = M
            for _ in range(args.n_rw - 1):
                M_power = M_power * M
                SE_rw.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw = torch.stack(SE_rw, dim=-1)

            g['stc_enc'] = SE_rw

    elif type_init == 'dg':
        for g in gs:
            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i, int(g_dg[i] - 1)] = 1

            g['stc_enc'] = SE_dg

    elif type_init == 'rw_dg':
        for g in gs:
            # SE_rw
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv = sp.diags(D)
            RW = A * Dinv
            M = RW

            SE = [torch.from_numpy(M.diagonal()).float()]
            M_power = M
            for _ in range(args.n_rw - 1):
                M_power = M_power * M
                SE.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw = torch.stack(SE, dim=-1)

            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i, int(g_dg[i] - 1)] = 1

            g['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1)
    return gs


def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (
            np.max(ano_score) - np.min(ano_score)))
    return ano_score

def csv_rw(path,temp_record):

    writer = csv.reader(open(path, "r"))
    # 获取数据集名
    dataset = temp_record[0]
    csv_record = list(writer)
    row_list = []
    # 获取数据集索引
    for row in csv_record:
        row_list.append(row[0])
    index_dataset = row_list.index(dataset)

    #获取auc值索引
    index_list = ["auc"]
    index_dict = {}
    for param in index_list:
        index_dict[param] = csv_record[0].index(param)
    csv_auc = float(csv_record[index_dataset][index_dict["auc"]])
    temp_auc = temp_record[index_dict["auc"]]
    if temp_auc > csv_auc:
        csv_record[index_dataset] = temp_record
        with open(path, "w", encoding="utf-8", newline="") as f:
            cur_writer = csv.writer(f)
            cur_writer.writerows(csv_record)
        print("Dataset:{} Update!".format(dataset))

def get_mean_loss(loss_list):
    all_elems = torch.cat(loss_list, dim=0)
    loss_mean = all_elems.mean()
    return loss_mean



def load_parameters_dict(params_file):
    if params_file is None:
        raise ValueError("params_file must not be None")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    resolved_path = params_file
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.join(base_dir, resolved_path)

    with open(resolved_path, "r", encoding="utf-8") as f:
        parameters_dict = json.load(f)

    if not isinstance(parameters_dict, dict):
        raise ValueError("parameters_dict loaded from JSON must be a dict")

    return parameters_dict



def set_args(args, cur_param):
    # args.mlp_lr = cur_param["mlp_lr"]
    args.lr = cur_param["lr"]
    args.hidden = cur_param["hidden"]
    args.enc_layer = cur_param["enc_layer"]
    args.dec_layer = cur_param["dec_layer"]
    # args.mlp_layer = cur_param["mlp_layer"]
    # args.k = cur_param["k"]

    args.alpha = cur_param["alpha"]
    args.beta = cur_param["beta"]

    args.batch_size = cur_param["batch_size"]
    if hasattr(args, 'num_clients'):
        args.num_clients = cur_param["num_clients"]
    if hasattr(args, 'cluster_num'):
        args.cluster_num = cur_param["cluster_num"]
    if hasattr(args, 'temperature'):
        args.temperature = cur_param["temperature"]

    args.local_epoch = cur_param["local_epoch"]

    return args


def prepare_data(dataset, gamma=1.0):
    labels = []
    labels_pos_weights = []

    for d in dataset:
        adj = to_dense_adj(d.edge_index,max_num_nodes=d.num_nodes)[0]
        n_nodes = adj.shape[0]
        adj = adj.flatten()
        pos_weight = (float(n_nodes * n_nodes - adj.sum()) / adj.sum()) ** gamma
        labels.append(adj)
        labels_pos_weights.append(pos_weight)

    return labels, labels_pos_weights


def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree

def split_data_ad(DS, idx, graphs, trained_label, percent, num_client):
    y = torch.cat([graph.y for graph in graphs])
    test_idx = []

    normal_class = trained_label
    abnormal_class = list(set(y.tolist()).difference(set([normal_class])))
    abnormal_num = []

    normal_class_idx = np.where(np.array(y.tolist()) == normal_class)
    normal_num = len(normal_class_idx[0])
    for ab in abnormal_class:
        abnormal_num.append(len(np.where(np.array(y.tolist()) == ab)[0]))

    train_sample_num = math.ceil(normal_num * percent)
    train_idx = random.sample(list(normal_class_idx[0]), train_sample_num)

    retain_train_idx = list(set(normal_class_idx[0]).difference(set(train_idx)))
    test_sample_num = min(min(abnormal_num), len(retain_train_idx))
    for ab in abnormal_class:
        temp_test_idx = np.where(np.array(y.tolist()) == ab)
        test_idx.extend(random.sample(list(temp_test_idx[0]), test_sample_num))
    test_idx.extend(retain_train_idx)

    os.makedirs(os.path.join('./data/TUDataset', DS), exist_ok=True)

    np.savetxt(
        './data/TUDataset/' + DS + '/test_idx_' + str(idx) + '_' + str(trained_label) + '_' + str(num_client) + '.txt',
        test_idx, fmt='%d')
    np.savetxt(
        './data/TUDataset/' + DS + '/train_idx_' + str(idx) + '_' + str(trained_label) + '_' + str(num_client) + '.txt',
        train_idx, fmt='%d')
    return np.array(train_idx).astype(dtype=int).tolist(), np.array(test_idx).astype(dtype=int).tolist()


def split_data_ad_multi(DS, graphs, trained_label, percent):
    y = torch.cat([graph.y for graph in graphs])
    test_idx = []
    normal_class = trained_label
    abnormal_class = list(set(y.tolist()).difference(set([normal_class])))
    abnormal_num = []
    normal_class_idx = np.where(np.array(y.tolist()) == normal_class)
    normal_num = len(normal_class_idx[0])
    for ab in abnormal_class:
        abnormal_num.append(len(np.where(np.array(y.tolist()) == ab)[0]))
    train_sample_num = math.ceil(normal_num * percent)
    train_idx = random.sample(list(normal_class_idx[0]), train_sample_num)

    retain_train_idx = list(set(normal_class_idx[0]).difference(set(train_idx)))

    test_sample_num = min(min(abnormal_num), len(retain_train_idx))
    for ab in abnormal_class:
        temp_test_idx = np.where(np.array(y.tolist()) == ab)
        test_idx.extend(random.sample(list(temp_test_idx[0]), test_sample_num))
    test_idx.extend(retain_train_idx)

    os.makedirs(os.path.join('./data/TUDataset', DS), exist_ok=True)
    np.savetxt(
        './data/TUDataset/' + DS + '/test_idx_' + str(normal_class) + '.txt',
        test_idx, fmt='%d')
    np.savetxt(
        './data/TUDataset/' + DS + '/train_idx_' + str(normal_class) + '.txt',
        train_idx, fmt='%d')
    return np.array(train_idx).astype(dtype=int).tolist(), np.array(test_idx).astype(dtype=int).tolist()


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes / numGraphs, numEdges / numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df


def obtain_avg_result(client_AUC, client_F1, DS):
    global hist_auc
    global hist_f1
    avg_AUC = np.around([np.mean(np.array(client_AUC))], decimals=4)
    avg_F1 = np.around([np.mean(np.array(client_F1))], decimals=4)
    if hist_auc < avg_AUC:
        hist_auc = avg_AUC
        hist_f1 = avg_F1
        AUCList = client_AUC
        F1_List = client_F1
        os.makedirs('./result', exist_ok=True)
        with open('./result/' + DS + '_result.txt', 'a') as f:
            f.write('Avg AUC:' + str(avg_AUC) + '\n')
            f.write('Clent AUC:' + str(AUCList) + '\n')
            f.write('Clent F1:' + str(F1_List) + '\n')

    return hist_auc, hist_f1

def init_metric():
    global hist_auc
    global hist_f1
    hist_auc = -1.
    hist_f1 = -1.


def init_structure_encoding_cached(args, gs, type_init, split_name,data,group="OneDataSet",data_type="multi"):
    cache_dir = f"./{data_type}_cache_structure"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{group}_{data}_{split_name}_{type_init}_rw{args.n_rw}_dg{args.n_dg}.pt")

    print(f"Computing structure encodings for {split_name} with type={type_init} ...")
    gs = init_structure_encoding(args, gs=gs, type_init=type_init,is_norm=False)

    # 保存计算结果
    torch.save([g['stc_enc'] for g in gs], cache_path)
    print(f"Structure encodings saved to {cache_path}")
    return gs