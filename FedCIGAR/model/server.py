import networkx as nx
import numpy as np
import random
import torch
from dtaidistance import dtw
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score



class Server():
    def __init__(self, autoEncoder, device):
        self.autoEncoder = autoEncoder.to(device)
        self.W_auto = {key: value for key, value in self.autoEncoder.named_parameters()}
        self.model_cache = []

        self.silence_round = 10
        self.window_len = 5
        self.ari_threshold = 0.5

        self.feat_is_activate = True
        self.feat_silence_round = self.silence_round
        self.feat_slide_window = []
        self.client_feat_list = []

    def reset(self):
        self.client_feat_list = []

    def feat_isActivate(self):

        if self.feat_is_activate == False:
            if self.feat_silence_round == 0:
                self.feat_is_activate = True
                self.feat_silence_round = self.silence_round
            else:
                self.feat_silence_round -= 1

        return self.feat_is_activate

    def feat_cluster(self, cluster_num):
        client_feat_all = torch.cat(self.client_feat_list, dim=0)

        if (len(self.feat_slide_window) == self.window_len):
            print("silience {} round".format(self.feat_silence_round))

            self.feat_is_activate = False
            # 获取滑动窗口 最后一次分簇信息
            unique = np.unique(self.feat_slide_window[-1])
            cluster_list = [np.where(self.feat_slide_window[-1] == lab)[0].tolist() for lab in unique]
            self.temp_feat_cluster = cluster_list
            self.feat_slide_window = []
            return self.temp_feat_cluster
        cluster_ids = KMeans(n_clusters=cluster_num, n_init=10).fit_predict(client_feat_all.cpu())

        cluster_ids = np.asarray(cluster_ids).astype(int)
        if (len(self.feat_slide_window) != 0):
            for temp_cluster in self.feat_slide_window:
                ari = adjusted_rand_score(cluster_ids, temp_cluster)
                if ari < self.ari_threshold:
                    self.feat_slide_window = []
                    break
        self.feat_slide_window.append(cluster_ids)
        print("slide_window_len :",len(self.feat_slide_window))
        unique = np.unique(cluster_ids)
        cluster_list = [np.where(cluster_ids == lab)[0].tolist() for lab in unique]

        return cluster_list

    def get_feat_slience_cluster(self):
        return self.temp_feat_cluster

    def aggregate_feat_cluster(self, client_clusters):

        for cluster in client_clusters:
            total_size = 0
            for client in cluster:
                total_size += client.train_size

            for k in self.W_auto.keys():
                self.W_auto[k].data = torch.div(torch.sum(
                    torch.stack([torch.mul(client.W_auto[k].data, client.train_size) for client in cluster]),
                    dim=0), total_size).clone()

            for client in cluster:
                with torch.no_grad():
                    for k in self.W_auto:
                        client.W_auto[k].data = self.W_auto[k].data.clone()



    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size

        for k in self.W_auto.keys():
            self.W_auto[k].data = torch.div(torch.sum(
                torch.stack([torch.mul(client.W_auto[k].data, client.train_size) for client in selected_clients]),
                dim=0), total_size).clone()

    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW_stu[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def compute_pairwise_distances(self, seqs, standardize=False):
        """ computes DTW distances """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances

    def min_cut(self, similarity, idc):
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            targs = []
            sours = []
            total_size = 0
            for client in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = client.W_stu[k]
                    dW[k] = client.dW_stu[k]
                targs.append(W)
                sours.append((dW, client.train_size))
                total_size += client.train_size
            # pass train_size, and weighted aggregate
            reduce_add_average(targets=targs, sources=sours, total_size=total_size)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW_stu[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW
        # return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW_stu[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.true_divide(torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-12)) + 1

    return angles.numpy()


def reduce_add_average(targets, sources, total_size):
    for target in targets:
        for name in target:
            tmp = torch.div(
                torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources]), dim=0),
                total_size).clone()
            target[name].data += tmp
