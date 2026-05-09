import os
import random
from random import choices

import pandas as pd
from numpy import loadtxt
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import add_self_loops

from Utils.utils import get_maxDegree, get_stats, get_numGraphLabels, split_data_ad, split_data_ad_multi, prepare_data, init_structure_encoding_cached,normalize_features
from model.client import Client_GC
from model.models import *
from model.server import Server


def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum / num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i * minSize:(i + 1) * minSize])

        for g in graphs[num_client * minSize:]:

            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks


def prepareData_oneDS(args,datapath, data, num_client, batchSize, percentage, convert_x=False, seed=None,
                      overlap=False):
    print('Client Num', num_client)
    normal_class = 0

    if data == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
    elif data == "MUTAG":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True,
                              transform=torch_geometric.transforms.remove_isolated_nodes.RemoveIsolatedNodes())
    elif data == "AIDS":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True,
                              transform=torch_geometric.transforms.remove_isolated_nodes.RemoveIsolatedNodes())
    elif data == "DD":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True)
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    graphs = [x for x in tudataset]
    print("  **", data, len(graphs))

    if data in ['IMDB-BINARY', 'MUTAG', 'DD', 'AIDS']:
        for i, g in enumerate(graphs):
            if g.x is not None:
                g.x = normalize_features(g.x)

    if data in ['IMDB-BINARY', 'MUTAG', 'DD', 'AIDS']:
        for i, g in enumerate(graphs):
            g.edge_index, _ = add_self_loops(g.edge_index, num_nodes=g.num_nodes)

    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features

    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        if not os.path.exists(
                './data/TUDataset/' + data + '/test_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                    num_client) + '.txt') or not os.path.exists(
            './data/TUDataset/' + data + '/train_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                num_client) + '.txt'):
            print('Split Data')
            train_idx, test_idx = split_data_ad(data, idx, ds_tvt, normal_class, percentage, num_client)
        else:
            train_idx = np.array(
                (loadtxt('./data/TUDataset/' + data + '/train_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                    num_client) + '.txt'))).astype(
                dtype=int).tolist()
            test_idx = np.array(
                (loadtxt('./data/TUDataset/' + data + '/test_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                    num_client) + '.txt'))).astype(
                dtype=int).tolist()
        print(train_idx)
        ds_train = [ds_tvt[i] for i in train_idx]
        ds_test = [ds_tvt[i] for i in test_idx]
        ds_val = ds_test

        train_labels, train_pos_w = prepare_data(ds_train)
        val_labels,   val_pos_w   = prepare_data(ds_val)
        test_labels,  test_pos_w  = prepare_data(ds_test)

        ds_train = init_structure_encoding_cached(args, gs=ds_train, type_init=args.type_init,
                                                      split_name="train",data=data,data_type="one")
        ds_test = init_structure_encoding_cached(args, gs=ds_test, type_init=args.type_init, split_name="test",data=data,data_type="one")
        ds_val = ds_test

        for d, lbl, pw in zip(ds_train, train_labels, train_pos_w):
            d.label      = lbl
            d.pos_weight = pw

        for d, lbl, pw in zip(ds_val, val_labels, val_pos_w):
            d.label      = lbl
            d.pos_weight = pw

        for d, lbl, pw in zip(ds_test, test_labels, test_pos_w):
            d.label      = lbl
            d.pos_weight = pw

        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=False)
        dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=False)
        num_graph_labels = get_numGraphLabels(ds_train)
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(ds_train))
        df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)

    return splitedData, df


def has_self_loops(edge_index):
    return torch.any(edge_index[0] == edge_index[1])


def prepareData_multiDS(args,datapath, normal_class, percentage, group='small', batchSize=32, convert_x=False, seed=None):
    assert group in ['molecules', 'molecules_tiny', 'small', 'mix', "mix_tiny", "biochem", "biochem_tiny", "socialnet"]

    if group == 'molecules' or group == 'molecules_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    if group == 'small':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]  # bioinformatics
    if group == 'mix' or group == 'mix_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS",  # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]  # social networks
    if group == 'biochem' or group == 'biochem_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]  # bioinformatics
    if group == 'socialnet':
        datasets = ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "COLLAB":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
        elif data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        elif data in ["MUTAG","AIDS"]:
            if group in ["biochem", "molecules","mix"]:
                tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr = True,transform = torch_geometric.transforms.remove_isolated_nodes.RemoveIsolatedNodes())
            else:
                tudataset = TUDataset(f"{datapath}/TUDataset", data)
                if convert_x:
                    maxdegree = get_maxDegree(tudataset)
                    tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
        else:
            if group in ["biochem", "molecules","mix"]:
                tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr = True)
                if convert_x:
                    maxdegree = get_maxDegree(tudataset)
                    tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
            else:
                tudataset = TUDataset(f"{datapath}/TUDataset", data)
                if convert_x:
                    maxdegree = get_maxDegree(tudataset)
                    tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))

        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))

        if group in ['small', 'molecules', 'biochem', 'mix']:
            for i,g in enumerate(graphs):
                if g.x is not None:
                    g.x = normalize_features(g.x)

        if group in ["small",'molecules', 'biochem', 'mix']:
            for i, g in enumerate(graphs):
                g.edge_index, _ = add_self_loops(g.edge_index, num_nodes=g.num_nodes)



        if not os.path.exists(
                './data/TUDataset/' + data + '/test_idx_' + str(
                    normal_class) + '.txt') or not os.path.exists(
            './data/TUDataset/' + data + '/train_idx_' + '_' + str(normal_class) + '.txt'):
            train_idx, test_idx = split_data_ad_multi(data, graphs, normal_class, percentage)

        else:
            train_idx = np.array(
                (loadtxt(
                    './data/TUDataset/' + data + '/train_idx_' + str(normal_class) + '.txt'))).astype(
                dtype=int).tolist()
            test_idx = np.array(
                (loadtxt(
                    './data/TUDataset/' + data + '/test_idx_' + str(normal_class) + '.txt'))).astype(
                dtype=int).tolist()
        graphs_train = [graphs[i] for i in train_idx]
        graphs_test = [graphs[i] for i in test_idx]
        graphs_val = graphs_test

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)

        train_labels, train_pos_w = prepare_data(graphs_train)
        val_labels,   val_pos_w   = prepare_data(graphs_val)
        test_labels,  test_pos_w  = prepare_data(graphs_test)

        graphs_train = init_structure_encoding_cached(args, gs=graphs_train, type_init=args.type_init,
                                                      split_name="train",data=data,group=group)
        graphs_test = init_structure_encoding_cached(args, gs=graphs_test, type_init=args.type_init, split_name="test",data=data,group=group)
        graphs_val = graphs_test

        for d, lbl, pw in zip(graphs_train, train_labels, train_pos_w):
            d.label      = lbl
            d.pos_weight = pw

        for d, lbl, pw in zip(graphs_val, val_labels, val_pos_w):
            d.label      = lbl
            d.pos_weight = pw

        for d, lbl, pw in zip(graphs_test, test_labels, test_pos_w):
            d.label      = lbl
            d.pos_weight = pw

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=False)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=False)

        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels, len(graphs_train))

        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)
    return splitedData, df


def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        if num_node_features == 0:
            num_node_features = 1

        share_model = AutoEncoder_Client(num_node_features,args.n_se, args.hidden, args.enc_layer, args.dec_layer,args.dropout,args.device)
        local_model = localWeight(num_node_features+args.n_se, args.hidden, 3,args.device,temp=args.temperature)
        model = nn.ModuleList([share_model,local_model])

        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        clients.append(
            Client_GC(model, idx, ds, train_size, dataloaders, optimizer, args))

    smode_autoEnc = AutoEncoder_Client(num_node_features,args.n_se, args.hidden, args.enc_layer, args.dec_layer,args.dropout,args.device)
    server = Server(smode_autoEnc, args.device)

    return clients, server, idx_clients


def setup_devices_multi(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        if num_node_features == 0:
            num_node_features = 1


        share_model = AutoEncoder_Client_Multi(num_node_features,args.n_se, args.hidden, args.enc_layer, args.dec_layer,args.dropout,args.device,args.gnn_type)
        local_model = localWeight(num_node_features + args.n_se, args.hidden, 3,args.device,args.gnn_type)
        model = nn.ModuleList([share_model,local_model])

        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        clients.append(
            Client_GC(model, idx, ds, train_size, dataloaders, optimizer, args))

    smode_autoEnc = AutoEncoder_Server_Multi(args.hidden,args.n_se, args.hidden, args.enc_layer, args.dec_layer,args.dropout,args.device,args.gnn_type)
    server = Server(smode_autoEnc, args.device)
    return clients, server, idx_clients
