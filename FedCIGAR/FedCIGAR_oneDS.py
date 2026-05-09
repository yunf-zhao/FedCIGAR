import argparse
import random
import warnings

import numpy as np
from Utils.utils import set_args, load_parameters_dict
from Utils import setupGC
from model.training import *

import os

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def fix_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=3,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')


    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--datapath', type=str, default='../LG-FGAD-main/data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='AIDS')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=5)

    parser.add_argument('--enc_layer', type=int, default=2,
                        help='Number of student GINconv layers')
    parser.add_argument('--dec_layer', type=int, default=2,
                        help='Number of student GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for inner solver;')

    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--alpha', type=float, default=0.5, metavar='N',
                        help='Weight of the mutual information loss')
    # DD--- 高beta
    parser.add_argument('--beta', type=float, default=0.1, metavar='N',
                        help='Weight of the mutual information loss')

    parser.add_argument('--n_rw', type=int, default=32,
                        help='Size of position encoding (random walk).')
    parser.add_argument('--n_dg', type=int, default=32,
                        help='Size of position encoding (max degree).')
    parser.add_argument('--type_init', help='the type of positional initialization',
                        type=str, default='rw_dg', choices=['rw', 'dg', 'rw_dg', 'ones'])
    parser.add_argument('--cluster_num', type=int, default=1)
    parser.add_argument('--temperature', type=int, default=1e2)

    parser.add_argument('--params_file', type=str, default='configs/params_oneDS.json',
                        help='Path to JSON file containing parameters_dict.')
    parser.add_argument('--datasets', nargs='+', default=["MUTAG"]
                        ,choices=["MUTAG","IMDB-BINARY","AIDS","DD"])


    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    parameters_dict = load_parameters_dict(args.params_file)

    seed_dataSplit = 123
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    percentage = 0.8
    dataset_list = args.datasets

    for dataset in dataset_list:

        args.data_group = dataset

        print("before num_clients :",args.num_clients)
        if args.data_group in parameters_dict:
            cur_param = parameters_dict[args.data_group]
            args = set_args(args,cur_param)
        else:
            if "small" in parameters_dict:
                cur_param = parameters_dict["small"]
                args = set_args(args, cur_param)
            else:
                raise KeyError(f"data_group '{args.data_group}' not found in parameters_dict and no 'small' fallback.")

        print("after num_clients :", args.num_clients)

        set_seed(0)
        splitedData, df_stats = setupGC.prepareData_oneDS(args,args.datapath, args.data_group,
                                                                     num_client=args.num_clients,
                                                                     batchSize=args.batch_size,
                                                                     percentage=percentage,
                                                                     convert_x=args.convert_x, seed=seed_dataSplit,
                                                                     overlap=args.overlap)
        hist_AUC = []
        hist_F1 = []
        print("Done")
        repNum = args.num_repeat

        # 运行10次
        for epoch in range(repNum):
            fix_random_seed(epoch)
            args.n_se = args.n_rw + args.n_dg

            init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
            print("\nDone setting up devices.")
            AUC, F1, = Run_FedCIGAR(init_clients, init_server, args.num_rounds, args.local_epoch, args.alpha, args.beta,
                              args.data_group, samp=None, cluster_num=args.cluster_num)
            hist_AUC.append(AUC)
            hist_F1.append(F1)

        Mean_AUC = np.around([np.mean(np.array(hist_AUC)), np.std(np.array(hist_AUC))], decimals=4)
        Mean_F1 = np.around([np.mean(np.array(hist_F1)), np.std(np.array(hist_F1))], decimals=4)
        print("-----" * 10)
        print("Dataset:", args.data_group)
        print('Average AUC: ' + str(np.around(Mean_AUC[0] * 100, 2)) + '±' + str(np.around(Mean_AUC[1] * 100, 2)))
        print('Average F1:  ' + str(np.around(Mean_F1[0] * 100, 2)) + '±' + str(np.around(Mean_F1[1] * 100, 2)))
        print("-----" * 10)