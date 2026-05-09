import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LeakyReLU
from torch_geometric.nn import GINConv, global_add_pool,GCNConv,global_mean_pool


class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""

    def __init__(self, w_d=1, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = 0

    def forward(self, f_s, f_t):

        # f_s/f_t : batch_size × feat_dim
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            # pdist 计算所有样本对之间的欧氏距离矩阵 t_d，形状 [B, B]，对角线置零。
            t_d = self.pdist(teacher, squared=False)
            # 距离的归一化处理
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        # 学生网络的归一化矩阵
        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        loss = self.w_d * loss_d  # + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res


class teacher_head(torch.nn.Module):
    def __init__(self, dim, device):
        super(teacher_head, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(dim * 4, dim * 3)
        self.fc2 = nn.Linear(dim * 3, dim * 2)
        self.fc3 = nn.Linear(dim * 2, dim)
        self.fc4 = nn.Linear(dim, 1)

    def forward(self, g_enc):
        x = F.leaky_relu(self.fc1(g_enc))
        x = F.leaky_relu(self.fc2(x))
        middle = F.leaky_relu(self.fc3(x))
        output = self.fc4(middle)
        return output


class student_head(torch.nn.Module):
    def __init__(self, dim, device):
        super(student_head, self).__init__()
        self.device = device
        # self.fc1 = nn.Linear(dim, int(dim/2))
        # self.fc2 = nn.Linear(int(dim/2), int(dim/4))
        # self.fc3 = nn.Linear(int(dim/4), 1)
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc1 = nn.Linear(dim, 16)
        # self.fc2 = nn.Linear(16, 8)
        # self.fc3 = nn.Linear(8, 1)

    def forward(self, g_enc):
        middle = F.leaky_relu(self.fc1(g_enc))
        middle = F.leaky_relu(self.fc2(middle))
        output = self.fc3(middle)
        return output


class Co_VGAE(nn.Module):
    def __init__(self, num_features, latent_dim, num_gc_layers, device, alpha=0.5, beta=1., gamma=.1):
        super(Co_VGAE, self).__init__()
        self.dataset_num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        # print(dataset_num_features)
        self.hidden_dim = latent_dim
        self.num_gc_layers = num_gc_layers
        self.embedding_dim = mi_units = self.hidden_dim * self.num_gc_layers
        self.base_gcn = Encoder(self.dataset_num_features, self.hidden_dim, self.num_gc_layers, self.device)
        self.gcn_mean = Encoder(self.embedding_dim, self.hidden_dim, self.num_gc_layers, self.device)
        self.gcn_logstddev = Encoder(self.embedding_dim, self.hidden_dim, self.num_gc_layers, self.device)
        self.decoder = Sequential(Linear(self.embedding_dim, math.ceil(self.dataset_num_features / 2)), LeakyReLU(),
                                  Linear(math.ceil(self.dataset_num_features / 2), self.dataset_num_features))

    def encode(self, x, edge_index, batch):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones(batch.shape[0]).to(self.device)

        # hidden --->节点级信息[n,num_gc_layers * dim]   --- 经过GINConv
        _, hidden = self.base_gcn(x, edge_index, batch)
        # uc = GINu(Xc,Ac)
        _, self.mean = self.gcn_mean(hidden, edge_index, batch)
        # log_sigma_c = GIN_sigma(Xc,Ac)
        _, self.logstd = self.gcn_logstddev(hidden, edge_index, batch)

        # 噪声数据
        gaussian_noise = torch.randn(batch.shape[0], self.hidden_dim * self.num_gc_layers).to(self.device)

        # zc_hat = uc + ε*exp(sigma_c)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, x, edge_index, batch):
        # 生成噪声特征 z
        z = self.encode(x, edge_index, batch)
        # 重构 x
        reconstructed_x = self.decoder(z)
        #  A = sigmoid(zzT)
        A_pred = dot_product_decode(z)
        return reconstructed_x, A_pred

    def latent_loss(self, z_mean, z_stddev):
        kl_divergence = 0.5 * torch.sum(torch.exp(z_stddev) + torch.pow(z_mean, 2) - 1. - z_stddev)
        return kl_divergence / z_mean.size(0)

    def loss(self, x_rec, x):
        reconstruction_loss = F.mse_loss(x_rec, x)
        kl_loss_node = self.latent_loss(self.mean, self.logstd)
        total_loss = reconstruction_loss + 0.001 * kl_loss_node
        return total_loss


class GraphConv(nn.Module):
    def __init__(self):
        super(GraphConv, self).__init__()

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        return x


class backbone_GIN(torch.nn.Module):
    def __init__(self, num_features, latent_dim, num_gc_layers, device):
        super(backbone_GIN, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device
        # self.nns = []
        # self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.nns = torch.nn.ModuleList()
        self.embedding_dim = latent_dim * num_gc_layers
        self.gin_conv = GraphConv()
        for i in range(self.num_gc_layers):
            bn = torch.nn.BatchNorm1d(latent_dim, eps=1e-04, affine=False, track_running_stats=True)
            if i:
                nn = Sequential(Linear(latent_dim, latent_dim), LeakyReLU(), Linear(latent_dim, latent_dim))
            else:
                nn = Sequential(Linear(num_features, latent_dim), LeakyReLU(), Linear(latent_dim, latent_dim))

            self.nns.append(nn)
            self.bns.append(bn)
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.fill_(0.0)
                torch.nn.init.xavier_uniform_(m.weight.data)
                # if m.bias is not None:
                #     m.bias.data.fill_(0.0)

    def forward(self, x, adj, batch):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        for i in range(self.num_gc_layers):
            # 聚合邻域信息 (GCN)
            x = self.gin_conv(x, adj)
            x = self.nns[i](x)
            x = self.bns[i](x)
            xs.append(x)
        xpool = [global_add_pool(x, batch) for x in xs]
        x_global = torch.cat(xpool, 1)
        x_node = torch.cat(xs, 1)
        g_enc = x_global
        l_enc = x_node
        return g_enc, l_enc


class Server_encoder_multi(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(Server_encoder_multi, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device
        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
            # nn = Sequential(Linear(dim, dim, bias=False))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim, eps=1e-04, affine=True, track_running_stats=True)

            self.convs.append(conv)
            self.bns.append(bn)





# class Encoder_multi(torch.nn.Module):
#     def __init__(self, num_features, dim, num_gc_layers, device):
#         super(Encoder_multi, self).__init__()
#         self.num_gc_layers = num_gc_layers
#         self.device = device
#         self.pre = torch.nn.Sequential(torch.nn.Linear(num_features, dim))
#         # self.nns = []
#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()
#         for i in range(num_gc_layers):
#             if i:
#                 nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
#             else:
#                 nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
#             # nn = Sequential(Linear(dim, dim, bias=False))
#             conv = GINConv(nn)
#             bn = torch.nn.BatchNorm1d(dim, eps=1e-04, affine=True, track_running_stats=True)
#
#             self.convs.append(conv)
#             self.bns.append(bn)
#
#     def forward(self, x, edge_index, batch):
#
#         if x is None or len(x.shape) == 1 or x.shape[1] == 0:
#             x = torch.ones((batch.shape[0], 1)).to(self.device)
#         xs = []
#         x = self.pre(x)
#         for i in range(self.num_gc_layers):
#             x = self.convs[i](x, edge_index)
#             x = self.bns[i](x)
#             xs.append(x)
#
#         xpool = [global_add_pool(x, batch) for x in xs]
#         x = torch.cat(xpool, 1)
#         return x, torch.cat(xs, 1)
#
#     def get_embeddings(self, loader):
#         ret = []
#         y = []
#         with torch.no_grad():
#             for data in loader:
#                 data.to(self.device)
#                 x, edge_index, batch = data.x, data.edge_index, data.batch
#                 if x is None:
#                     x = torch.ones((batch.shape[0], 1)).to(self.device)
#                 x, _ = self.forward(x, edge_index, batch)
#                 ret.append(x.cpu().numpy())
#                 y.append(data.y.cpu().numpy())
#         ret = np.concatenate(ret, 0)
#         y = np.concatenate(y, 0)
#         return ret, y


class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

# class Encoder(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(Encoder, self).__init__()
#
#         self.gc1 = GCNConv(nfeat, nhid)
#         self.gc2 = GCNConv(nhid, nhid)
#         self.norm = torch.nn.BatchNorm1d(nhid)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = self.norm(self.gc1(x, adj))
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.gc2(x, adj))
#
#         return x
#
# class Decoder(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(Decoder, self).__init__()
#
#         self.gc1 = GCNConv(nhid, nhid)
#         self.gc2 = GCNConv(nhid, nfeat)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.gc2(x, adj))
#
#         return x


class localWeight(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device,gnn_type="gcn",temp=1):
        super(localWeight, self).__init__()
        self.temp = temp

        self.num_gc_layers = num_gc_layers
        self.device = device

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        # self.dropout_layer = torch.nn.Dropout(p=self.dropout)

        for i in range(num_gc_layers):
            # 1.
            if i:
                in_dim = dim
            else:
                in_dim = num_features

            if i != num_gc_layers-1:
                 out_dim = dim
            else:
                out_dim = 1

            if(gnn_type=="gin"):
                gin_layer = Sequential(Linear(in_dim, dim), torch.nn.Sigmoid(), Linear(dim, out_dim))
                conv = GINConv(gin_layer)
            elif(gnn_type=="gcn"):
                conv = GCNConv(in_dim, out_dim)

            # 1.GIN卷积
            # gin_layer = Sequential(Linear(in_dim, dim), torch.nn.Sigmoid(), Linear(dim, dim))
            # conv = GINConv(gin_layer)

            # 2.GCN卷积
            # conv = GCNConv(in_dim,out_dim)

            # act = nn.ReLU()
            act = nn.Sigmoid()
            # act = nn.LeakyReLU(inplace=True)
            bn = torch.nn.BatchNorm1d(out_dim, eps=1e-04, affine=True, track_running_stats=True)
            self.convs.append(conv)
            self.acts.append(act)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):

        eps = 1e-6

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []

        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            if(i==self.num_gc_layers-1):
                x = x/self.temp

            x = self.acts[i](x)

            # if(i!=self.num_gc_layers-1):
            #     x = F.dropout(x,self.dropout,training=self.training)
            xs.append(x)

        # min - max 归一化
        # ret = (xs[-1] - xs[-1].min()) / (xs[-1].max() - xs[-1].min() + 1e-6)

        # 1.返回特征列表
        return xs[-1] + eps
        # 2.返回concat特征
        # return x,emb_x


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers,dropout, device,gnn_type="gcn"):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.dropout = dropout
        self.bns = torch.nn.ModuleList()
        # self.dropout_layer = torch.nn.Dropout(p=self.dropout)

        for i in range(num_gc_layers):
            # 1.
            if i:
                in_dim = dim
            else:
                in_dim = num_features

            if(gnn_type=="gin"):
                # 1.GIN卷积
                gin_layer = Sequential(Linear(in_dim, dim), torch.nn.Sigmoid(), Linear(dim, dim))
                conv = GINConv(gin_layer)

            # 2.GCN卷积
            elif (gnn_type == "gcn"):
                conv = GCNConv(in_dim,dim)

            # act = nn.ReLU()
            act = nn.Sigmoid()
            # act = nn.LeakyReLU(inplace=True)
            bn = torch.nn.BatchNorm1d(dim, eps=1e-04, affine=True, track_running_stats=True)
            self.convs.append(conv)
            self.acts.append(act)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []

        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            x = self.acts[i](x)
            x = F.dropout(x, self.dropout, training=self.training)

            # if(i!=self.num_gc_layers-1):
            #     x = F.dropout(x,self.dropout,training=self.training)
            x = self.bns[i](x)
            xs.append(x)

        xpool = global_mean_pool(xs[-1], batch)

        # 1.返回特征列表
        return xpool,xs[-1]
        # 2.返回concat特征
        # return x,emb_x


class Muti_Pre(torch.nn.Module):
    def __init__(self, num_features, dim,  device):
        super(Muti_Pre, self).__init__()
        self.device = device
        self.pre = torch.nn.Sequential(torch.nn.Linear(num_features, dim))

    def forward(self, x, batch):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        x = x.to(self.device)
        x = self.pre(x)
        return x


class Encoder_multi(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(Encoder_multi, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.device = device

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            conv = GCNConv(dim,dim)
            act = nn.LeakyReLU(inplace=True)

            bn = torch.nn.BatchNorm1d(dim, eps=1e-04, affine=True, track_running_stats=True)

            self.convs.append(conv)
            self.acts.append(act)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        # 每一层做 GINConv(两个MLP + LeakyReLU) + BatchNorm1d
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            x = self.acts[i](x)
            x = self.bns[i](x)
            xs.append(x)

        # global_add_pool : 对拼接到一起的所有节点h,根据batch向量 把节点分组,然后对每个组的节点特征做「按维度求和」。
        # 最终得到一个 [B, dim] 的张量
        xpool = [global_mean_pool(x, batch) for x in xs]

        # x : 把所有层的 [B, dim] 池化结果在特征维度上串联，得到 [B, num_gc_layers * dim]  ---> 图级信息
        # xs : 把各层卷积后的节点表示 xs[i]（形状 [N, dim]）同样在特征维度上串起来，得到 [N, num_gc_layers * dim]  ---> 节点级信息
        x = torch.cat(xpool, 1)

        # 1.返回特征列表
        # return xpool,xs[-1]
        # 2.返回concat特征
        return x,xs[-1]


# class Attribute_Decoder(nn.Module):
#     def __init__(self, num_features, dim, num_gc_layers,dropout,device):
#         super(Attribute_Decoder, self).__init__()
#
#         self.lin1 = torch.nn.Linear(in_dim, hid_dim)
#         self.lin2 = torch.nn.Linear(hid_dim, out_dim)
#
#     def forward(self, x):
#         x = torch.relu(self.lin1(x))
#         x = self.lin2(x)
#
#         return x

class Attribute_Decoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers,dropout,device,gnn_type="gcn",enc_layer=1):
        super(Attribute_Decoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device
        self.dropout = dropout

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lines = torch.nn.ModuleList()

        in_dim = enc_layer * dim

        for i in range(num_gc_layers):
            out_dim = dim if i != num_gc_layers-1 else num_features
            enc_dim = dim if i !=0 else in_dim

            line = torch.nn.Sequential(torch.nn.Linear(enc_dim, out_dim))

            # if分支
            if(gnn_type=="gin"):
                # 1.GIN卷积
                gin_layer = Sequential(Linear(enc_dim, dim), torch.nn.Sigmoid(), Linear(dim, out_dim))
                conv = GINConv(gin_layer)

            # 2.GCN卷积
            elif (gnn_type == "gcn"):
                conv = GCNConv(enc_dim,out_dim)

            # 1.GCN卷积
            # conv = GCNConv(enc_dim,out_dim)

            # 2.GIN卷积
            # gin_layer = Sequential(Linear(enc_dim, dim), torch.nn.Sigmoid(), Linear(dim, out_dim))
            # conv = GINConv(gin_layer)

            # act = nn.ReLU()
            # act = nn.LeakyReLU(inplace=True)
            act = nn.Sigmoid()
            bn = torch.nn.BatchNorm1d(out_dim, eps=1e-04, affine=True, track_running_stats=True)

            self.convs.append(conv)
            self.acts.append(act)
            self.lines.append(line)
            self.bns.append(bn)

    def forward(self, x, edge_index):

        xs = []

        # 每一层做 GINConv(两个MLP + LeakyReLU) + BatchNorm1d
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            # x = self.acts[i](x)
            x = F.dropout(x,self.dropout,training=self.training)

            if(i!=self.num_gc_layers-1):
                x = self.acts[i](x)

            # if(i!=self.num_gc_layers-1):
            #     x = F.dropout(x, self.dropout, training=self.training)
            x = self.bns[i](x)
            xs.append(x)


        return xs[-1]


class Structure_Decoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers,dropout,device,gnn_type="gcn",enc_layer=1):
        super(Structure_Decoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device

        self.dropout = dropout
        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lines = torch.nn.ModuleList()

        in_dim = enc_layer * dim

        for i in range(num_gc_layers):
            out_dim = dim if i != num_gc_layers-1 else num_features
            enc_dim = dim if i !=0 else in_dim

            line = torch.nn.Sequential(torch.nn.Linear(enc_dim, out_dim))

            # if分支
            if(gnn_type=="gin"):
                # 1.GIN卷积
                gin_layer = Sequential(Linear(enc_dim, dim), torch.nn.Sigmoid(), Linear(dim, out_dim))
                conv = GINConv(gin_layer)

            # 2.GCN卷积
            elif (gnn_type == "gcn"):
                conv = GCNConv(enc_dim,out_dim)

            # 1.GCN卷积
            # conv = GCNConv(enc_dim,out_dim)

            # 2.GIN卷积
            # gin_layer = Sequential(Linear(enc_dim, dim), torch.nn.Sigmoid(), Linear(dim, out_dim))
            # conv = GINConv(gin_layer)

            # act = nn.ReLU()
            # act = nn.LeakyReLU(inplace=True)
            act = nn.Sigmoid()
            bn = torch.nn.BatchNorm1d(out_dim, eps=1e-04, affine=True, track_running_stats=True)

            self.convs.append(conv)
            self.acts.append(act)
            self.lines.append(line)
            self.bns.append(bn)

    def forward(self, x, edge_index):

        # print(x.shape)

        xs = []

        # # 每一层做 GINConv(两个MLP + LeakyReLU) + BatchNorm1d
        for i in range(self.num_gc_layers):
            if(i!=self.num_gc_layers-1):
                x = self.convs[i](x, edge_index)
                # x = self.lines[i](x)
                x = self.acts[i](x)
                # x = self.acts[i](x)
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = self.convs[i](x, edge_index)
                # x = self.lines[i](x)
                x = torch.sigmoid(x / 2)
            x = self.bns[i](x)
            xs.append(x)

        # for i in range(self.num_gc_layers):
        #     x = self.convs[i](x, edge_index)
        #     x = self.acts[i](x)
        #     x = F.dropout(x,self.dropout,training=self.training)
        #
        #     # if(i!=self.num_gc_layers-1):
        #     #     x = self.acts[i](x)
        #
        #     # if(i!=self.num_gc_layers-1):
        #     #     x = F.dropout(x, self.dropout, training=self.training)
        #     x = self.bns[i](x)
        #     xs.append(x)


        return xs[-1]


# 掩码自编码器
class MaskAutoEncoder(torch.nn.Module):
    def __init__(self, feat_size, hidden_size, enc_layers,dec_layers,dropout,device):
        super(MaskAutoEncoder, self).__init__()
        self.enc_mask_token = nn.Parameter(torch.zeros(1, feat_size))

        self.enc_mask_token = nn.Parameter(torch.zeros(1, feat_size))
        self.encoder = Encoder(feat_size, hidden_size,enc_layers,dropout,device)
        # input_size = hidden_size * enc_layers

        # 1.Encoder concat 1-l
        # self.attr_decoder = Attribute_Decoder(feat_size, hidden_size,dec_layers,dropout,device,enc_layers)
        # self.struct_decoder = Structure_Decoder(feat_size, hidden_size, dec_layers,dropout, device,enc_layers)

        # 2.Encoder
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size,dec_layers,dropout,device)
        self.struct_decoder = Structure_Decoder(feat_size, hidden_size, dec_layers,dropout, device)

    def forward(self, x, edge_index ,batch):

        x_pool,x_enc = self.encoder(x,edge_index, batch)
        x_attDec = self.attr_decoder(x_enc,edge_index)
        x_struDec = self.struct_decoder(x_enc,edge_index)

        return x_struDec,x_attDec,x_enc



# 原自编码器
class AutoEncoder_Server_Multi(torch.nn.Module):
    def __init__(self, feat_size,struct_size, hidden_size, enc_layers,dec_layers,dropout,device,gnn_type):
        super(AutoEncoder_Server_Multi, self).__init__()
        self.attr_encoder = Encoder(hidden_size, hidden_size,enc_layers,dropout,device,gnn_type)
        self.struct_encoder = Encoder(hidden_size, hidden_size,enc_layers,dropout,device,gnn_type)

        self.e_to_d = torch.nn.Linear(2 * hidden_size, hidden_size)
        # input_size = hidden_size * enc_layers

        # 1.Encoder concat 1-l
        # self.attr_decoder = Attribute_Decoder(feat_size, hidden_size,dec_layers,dropout,device,enc_layers)
        # self.struct_decoder = Structure_Decoder(feat_size, hidden_size, dec_layers,dropout, device,enc_layers)

        # 2.Encoder
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size,dec_layers,dropout,device,gnn_type)
        self.struct_decoder = Structure_Decoder(feat_size, hidden_size, dec_layers,dropout, device,gnn_type)


class AutoEncoder_Client_Multi(torch.nn.Module):
    def __init__(self, feat_size,struct_size, hidden_size, enc_layers,dec_layers,dropout,device,gnn_type='gcn'):
        super(AutoEncoder_Client_Multi, self).__init__()
        self.device = device
        self.pre = torch.nn.Linear(feat_size, hidden_size)
        self.embedding_s = torch.nn.Linear(struct_size, hidden_size)

        self.attr_last_layer = torch.nn.Sequential(torch.nn.Linear(hidden_size, feat_size))

        self.attr_encoder = Encoder(hidden_size, hidden_size,enc_layers,dropout,device,gnn_type)
        self.struct_encoder = Encoder(hidden_size, hidden_size,enc_layers,dropout,device,gnn_type)
        self.e_to_d = torch.nn.Linear(2 * hidden_size, hidden_size)


        # 2. Emb don't concat
        self.attr_decoder = Attribute_Decoder(hidden_size  , hidden_size,dec_layers,dropout,device,gnn_type)
        self.struct_decoder = Structure_Decoder(hidden_size, hidden_size, dec_layers,dropout, device,gnn_type)



    def forward(self, x,s, edge_index ,batch):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        x = x.to(self.device)
        s = s.to(self.device)
        x = self.pre(x)
        s = self.embedding_s(s)

        x_pool,x_enc = self.attr_encoder(x,edge_index, batch)
        s_pool,s_enc = self.struct_encoder(s,edge_index, batch)

        emb_pool = torch.cat((x_pool,s_pool),-1)

        emb = torch.cat((x_enc,s_enc),-1)
        emb = self.e_to_d(emb)

        x_attDec = self.attr_decoder(emb,edge_index)
        x_struDec = self.struct_decoder(emb,edge_index)

        x_attDec = self.attr_last_layer(x_attDec)

        return x_struDec,x_attDec,emb,emb_pool


# 原自编码器
class AutoEncoder_Server(torch.nn.Module):
    def __init__(self, feat_size,struct_size, hidden_size, enc_layers,dec_layers,dropout,device,gnn_type='gcn'):
        super(AutoEncoder_Server, self).__init__()
        self.attr_encoder = Encoder(feat_size, hidden_size,enc_layers,dropout,device,gnn_type)
        self.struct_encoder = Encoder(struct_size, hidden_size,enc_layers,dropout,device,gnn_type)
        self.e_to_d = torch.nn.Linear(2 * hidden_size, hidden_size)        # input_size = hidden_size * enc_layers

        # 1.Encoder concat 1-l
        # self.attr_decoder = Attribute_Decoder(feat_size, hidden_size,dec_layers,dropout,device,enc_layers)
        # self.struct_decoder = Structure_Decoder(feat_size, hidden_size, dec_layers,dropout, device,enc_layers)

        # 2.Encoder
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size,dec_layers,dropout,device)
        self.struct_decoder = Structure_Decoder(struct_size, hidden_size, dec_layers,dropout, device)

    def forward(self, x,s, edge_index ,batch):
        print("="*20)
        print("attr:",x.shape)
        print(self.attr_encoder)
        print("struct: ",s.shape,s)
        print(self.struct_encoder)

        x_pool,x_enc = self.attr_encoder(x,edge_index, batch)

        s_pool,s_enc = self.struct_encoder(s,edge_index, batch)



        emb = torch.cat((x_enc,s_enc),-1)
        emb = self.e_to_d(emb)

        x_attDec = self.attr_decoder(emb,edge_index)
        x_struDec = self.struct_decoder(emb,edge_index)

        return x_struDec,x_attDec,x_enc,x_pool

# 原自编码器
class AutoEncoder_Client(torch.nn.Module):
    def __init__(self, feat_size,struct_size, hidden_size, enc_layers,dec_layers,dropout,device):
        super(AutoEncoder_Client, self).__init__()
        self.attr_encoder = Encoder(feat_size, hidden_size,enc_layers,dropout,device)
        self.struct_encoder = Encoder(struct_size, hidden_size,enc_layers,dropout,device)
        self.e_to_d = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.device = device
        # input_size = hidden_size * enc_layers

        # 1.Encoder concat 1-l
        # self.attr_decoder = Attribute_Decoder(feat_size, hidden_size,dec_layers,dropout,device,enc_layers)
        # self.struct_decoder = Structure_Decoder(feat_size, hidden_size, dec_layers,dropout, device,enc_layers)

        # 2.Encoder
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size,dec_layers,dropout,device)
        self.struct_decoder = Structure_Decoder(struct_size, hidden_size, dec_layers,dropout, device)

    def forward(self, x,s, edge_index ,batch):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)

        x = x.to(self.device)
        s = s.to(self.device)

        x_pool,x_enc = self.attr_encoder(x,edge_index, batch)
        s_pool, s_enc = self.struct_encoder(s, edge_index, batch)

        emb = torch.cat((x_enc,s_enc),-1)
        emb = self.e_to_d(emb)

        x_attDec = self.attr_decoder(emb,edge_index)
        x_struDec = self.struct_decoder(emb,edge_index)

        return x_struDec,x_attDec,x_enc,x_pool




class Muti_Pre(torch.nn.Module):
    def __init__(self, num_features, dim,  device):
        super(Muti_Pre, self).__init__()
        self.device = device
        self.pre = torch.nn.Sequential(torch.nn.Linear(num_features, dim))

    def forward(self, x, batch):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        x = x.to(self.device)
        x = self.pre(x)
        return x



# class Server_AutoEncoder_multi(torch.nn.Module):
#     def __init__(self, feat_size, hidden_size, enc_layers, dec_layers, device):
#         super(Server_AutoEncoder_multi, self).__init__()
#
#         self.encoder = Encoder(feat_size, hidden_size, enc_layers, device)
#         self.decoder = Decoder(feat_size, hidden_size, dec_layers, device)
#         # self.pre = torch.nn.Sequential(torch.nn.Linear(feat_size, hidden_size))

class CrossAttn(nn.Module):
    def __init__(self, embedding_dim):
        super(CrossAttn, self).__init__()
        self.embedding_dim = embedding_dim

        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query_X, support_X):
        # H_hat = softmax(QK(T)/de)Hk

        Q = self.Wq(query_X)  # query
        K = self.Wk(support_X)  # key
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(
            torch.tensor(self.embedding_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_query_embeddings = torch.matmul(attention_weights, support_X)

        # print(" any query_X nan?", torch.isnan(query_X).any())
        # print(" any support_X nan?", torch.isnan(support_X).any())
        # print(" any Wq weight nan?", torch.isnan(self.Wq.weight).any())
        # print(" any Wk weight nan?", torch.isnan(self.Wk.weight).any())
        # print("query_X  size:", query_X.size())
        # print("support_X  size:", support_X.size())
        # print("query_X:", query_X)
        # print("support_X:", support_X)
        # print("Q:", Q)
        # print("K:", K)

        return weighted_query_embeddings




def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)
