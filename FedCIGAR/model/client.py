import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from Utils.compute_metric import compute_pre_recall_f1
from Utils.utils import normalize_score
import torch.nn.functional as F



class Client_GC():

    def __init__(self,model, client_id, client_name, train_size, dataLoader,optimizer, args):

        self.autoEnc = model[0].to(args.device)
        self.localModel = model[1].to(args.device)
        self.device = args.device

        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args

        self.W_auto = {key: value for key, value in self.autoEnc.named_parameters()}
        self.dW_auto = {key: torch.zeros_like(value) for key, value in self.autoEnc.named_parameters()}
        self.W_old_auto = {key: value.data.clone() for key, value in self.autoEnc.named_parameters()}

        self.gconvNames = None

        self.gconvNames_stu = None
        self.gconvNames_sharedGIN = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def get_Encoder_Emb(self):
        feat_input = self.feat_emb

        feat_mean = feat_input.mean(dim=0, keepdim=True)
        feat_min = feat_input.min(dim=0).values.unsqueeze(0)
        feat_max = feat_input.max(dim=0).values.unsqueeze(0)

        client_feat = torch.cat([feat_mean,feat_min,feat_max],dim=1)
        return client_feat

    def send_Emd(self,server):
        client_feat = self.get_Encoder_Emb()
        server.client_feat_list.append(client_feat)

    def download_from_server(self, server):
        with torch.no_grad():
            for k in server.W_auto:
                self.W_auto[k].data = server.W_auto[k].data.clone()

    def evaluate(self,alpha,beta):
        return eval_gc(self.autoEnc,self.localModel, self.dataLoader['test'],
                       self.args.device,alpha,beta)

    def local_train(self, local_epoch, alpha,beta):

        loss,feat_emb = train_gc(self.autoEnc,self.localModel,
                               self.dataLoader, self.optimizer, local_epoch,
                               self.args.device, alpha,beta)
        self.feat_emb = feat_emb

        return  loss


def train_gc(autoEnc,localmodel, dataloaders, optimizer,
             local_epoch, device, alpha, beta):

    train_loader, test_loader = dataloaders['train'], dataloaders['test']

    loss_ret = []
    feat_emb_list = []

    for epoch in range(local_epoch):
        autoEnc.train()
        localmodel.train()

        epoch_loss = 0.
        for _, batch in enumerate(train_loader):
            data = batch.to(device)

            x, edge_index, batch,label = data.x, data.edge_index, data.batch,data.label
            ptr,struct_pos = data.ptr,data.pos_weight
            s = data.stc_enc
            sizes = [(ptr[i + 1] - ptr[i]) ** 2 for i in range(len(ptr) - 1)]

            labels_per_graph = label.split(sizes, dim=0)

            optimizer.zero_grad()

            # 自编码器重构
            A_hat_all, X_hat_all,x_enc,x_pool  = autoEnc(x,s, edge_index, batch)
            x_s_cat = torch.cat((x, s), dim=1)
            pos_all = localmodel(x_s_cat, edge_index, batch)

            if epoch == local_epoch-1:
                feat_emb_list.append(x_pool)

            loss_1_all,loss_2_all = 0.,0.



            for graph_id in range(len(ptr) - 1):
                start, end = ptr[graph_id], ptr[graph_id + 1]
                # 原始特征
                X = x[start:end]
                # 重构特征
                X_hat = X_hat_all[start:end]
                A_hat = A_hat_all[start:end]
                cur_pos = pos_all[start:end]

                A_flatten = torch.matmul(A_hat, A_hat.T).flatten()
                cur_label = labels_per_graph[graph_id]

                cur_struct_pos = struct_pos[graph_id].to(device)

                bce_loss = F.binary_cross_entropy_with_logits(
                    A_flatten,
                    cur_label.float(),
                    weight=cur_struct_pos,
                    reduction='none'
                )
                n = len(cur_pos)
                pos_weight = cur_pos.view(-1)

                pos_weight_matrix = pos_weight.view(n, 1) * pos_weight.view(1, n)  # 维度为 (n, n)
                pos_weight_expanded = pos_weight_matrix.flatten()
                weighted_L1 = torch.mean(bce_loss * pos_weight_expanded)  + beta * torch.std(bce_loss * pos_weight_expanded)

                feat_sim = 1 - F.cosine_similarity(X, X_hat, dim=1, eps=1e-6)
                weighted_L2 = torch.mean(feat_sim * pos_weight) + beta * torch.std(feat_sim * pos_weight)


                loss_1_all += weighted_L1
                loss_2_all += weighted_L2

                cur_loss = (1-alpha)*weighted_L1 + alpha*weighted_L2
                loss_ret.append(cur_loss)

            loss_1_all = loss_1_all/(len(ptr)-1)
            loss_2_all = loss_2_all/(len(ptr)-1)

            loss = (1-alpha)*loss_1_all + (alpha)*loss_2_all
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()

    loss_ret = torch.tensor(loss_ret)

    feat_emb = [torch.tensor(x, dtype=torch.float32) for x in feat_emb_list]
    feat_emb = torch.cat(feat_emb, dim=0)

    return loss_ret,feat_emb


def eval_gc(autoEnc,localmodel,test_loader, device,alpha,beta):


    with torch.no_grad():
        label_list = []
        score_list = []

        autoEnc.eval()
        localmodel.eval()

        for batch in test_loader:
            data = batch.to(device)
            ptr,struct_pos = data.ptr,data.pos_weight
            s = data.stc_enc

            sizes = [(ptr[i + 1] - ptr[i]) ** 2 for i in range(len(ptr) - 1)]
            x, edge_index, batch,adj_label = data.x, data.edge_index, data.batch,data.label
            labels_per_graph = adj_label.split(sizes, dim=0)

            A_hat_all, X_hat_all,_,_ = autoEnc(x, s,edge_index, batch)
            x_s_cat = torch.cat((x, s), dim=1)
            pos_all = localmodel(x_s_cat, edge_index, batch)

            graph_labels = data.y  # shape: [num_graphs]
            for graph_id in range(len(ptr) - 1):
                start, end = ptr[graph_id], ptr[graph_id + 1]
                X = x[start:end]
                X_hat = X_hat_all[start:end]
                A_hat = A_hat_all[start:end]

                cur_pos = pos_all[start:end]
                A_flatten = torch.matmul(A_hat, A_hat.T).flatten()
                cur_label = labels_per_graph[graph_id]


                bce_loss = F.binary_cross_entropy_with_logits(
                    A_flatten,
                    cur_label.float(),
                    reduction='none'
                )


                n = len(cur_pos)
                pos_weight = cur_pos.view(-1)

                pos_weight_matrix = pos_weight.view(n, 1) * pos_weight.view(1, n)
                pos_weight_expanded = pos_weight_matrix.flatten()
                weighted_L1 = torch.mean(bce_loss * pos_weight_expanded) + beta * torch.std(bce_loss * pos_weight_expanded)

                feat_sim = F.cosine_similarity(X, X_hat, dim=1, eps=1e-6)
                weighted_L2 = torch.mean(feat_sim * pos_weight) - beta * torch.std(feat_sim * pos_weight)

                cur_loss = (alpha - 1) * weighted_L1 + (alpha) * weighted_L2

                score_list.append(cur_loss)
                cur_graph_label = graph_labels[graph_id].float()
                label_list.append(cur_graph_label)

        labels = torch.tensor(label_list)
        scores = torch.tensor(score_list)

        labels = labels.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        scores = normalize_score(scores)

        labels = np.where(labels == 0, 1, 0)

        auc = roc_auc_score(labels,scores)

        test_f1, test_recall = compute_pre_recall_f1(labels, scores)
    return auc,test_f1