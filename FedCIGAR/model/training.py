import torch
from Utils.utils import obtain_avg_result, init_metric
import torch.nn.functional as F
from time import perf_counter

def Run_FedCIGAR(clients, server, COMMUNICATION_ROUNDS, local_epoch, alpha,beta, DS, samp=None,cluster_num=2,
                 early_stop_patience=15):
    sever_cluster_num = cluster_num

    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    init_metric()

    best_auc = -float('inf')
    rounds_no_improve = 0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):

        start_time = perf_counter()

        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        selected_clients = clients

        client_loss_list =[]
        feat_state = server.feat_isActivate()
        if feat_state:
            server.reset()

        for client in selected_clients:
            loss = client.local_train(local_epoch, alpha,beta)
            client_loss_list.append(loss)

            if feat_state:
                client.send_Emd(server)

        if feat_state:
            feat_cluster_indices = server.feat_cluster(sever_cluster_num)
        else:
            feat_cluster_indices = server.get_feat_slience_cluster()
        client_feat_cluster = [[clients[i] for i in idcs] for idcs in feat_cluster_indices]
        print("cluster : ",feat_cluster_indices)
        server.aggregate_feat_cluster(client_feat_cluster)


        client_AUC = []
        client_F1 = []
        for client in clients:
            test_auc, test_f1 = client.evaluate(alpha,beta)

            client_AUC.append(test_auc)
            client_F1.append(test_f1)
        avg_AUC, avg_F1 = obtain_avg_result(client_AUC, client_F1, DS)

        end_time = perf_counter()

        print("="*20,c_round,"="*20)
        print("Dataset",DS,'conmmunication rounds:', c_round,"AUC:",avg_AUC,"F1",avg_F1,"Time",end_time-start_time)

        avg_AUC = float(avg_AUC)
        avg_F1 = float(avg_F1)
        if avg_AUC > best_auc:
            best_auc = avg_AUC
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
            print(f"No improvement for {rounds_no_improve} rounds (best AUC: {best_auc:.4f}).")
            if rounds_no_improve >= early_stop_patience:
                print(f"Early stopping triggered. No AUC improvement in {early_stop_patience} rounds.")
                break
    return avg_AUC, avg_F1


