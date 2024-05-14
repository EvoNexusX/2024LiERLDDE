from __future__ import absolute_import, division, print_function
import os
import argparse
import torch
from math import log,sqrt
from model import ES
from train import train_loop, render_env

parser = argparse.ArgumentParser(description='ERL-DDE')
parser.add_argument('--restore', nargs='?', const=True, default=False,
                    help='If specified, restore model from the latest checkpoint.')
parser.add_argument('--silent', action='store_true',
                    help='Keep training precess silent')
parser.add_argument('--test', action='store_true',
                    help='Test env')



if __name__ == '__main__':
    args = parser.parse_args()
    # Creating folder if training
    args.env_name = "BACAP"
    if not args.test:
        for i in range(1000):
            folder_name = str(i)
            folder_name = os.path.join(args.env_name+"_checkpoints", folder_name)
            folder_path = os.path.join(os.getcwd(), folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print("Create folder", folder_path)
                break
        args.folder_path = folder_path
    # Initializing training parameters
    args.device = torch.device("cpu")
    args.train_loop = 1
    args.max_episode_length = 1000
    args.train_folder = r"./case_code"
    args.test_case = r"./case_code/test.txt"
    args.train_case_num = 200
    # Initializing algorithm parameters
    args.n = ES().count_parameters() # Searching dimension
    args.para_number = args.n
    args.r1 = 0.5
    args.r2 = 0.2
    args.h = sqrt(args.n)
    args.cluster_n = 5
    args.lambda_ = 80
    args.need_restart = [False for _ in range(args.cluster_n)]
    args.miu_0 = args.lambda_ // 2
    args.miu = args.miu_0
    args.sum_miu = sum([log(j) for j in range(1, args.miu + 1)])
    args.w = [(log((args.miu + 1) / i)) / (args.miu * log(args.miu + 1) - args.sum_miu) for i in range(1, args.miu + 1)]
    args.miu_eff = 1 / (sum([w * w for w in args.w]))
    args.c_cov = 1 / (3 * sqrt(args.n) + 5)
    args.c = 1 / (3 * sqrt(args.n) + 5)
    args.q_ = 0.3
    args.c_s = 0.3
    args.d_sig = 1
    args.sigma = [0.05 for _ in range(args.cluster_n)]
    args.p = [[] for _ in range(args.cluster_n)]
    args.s = [0 for _ in range(args.cluster_n)]
    args.lr = 1
    synced_model = [ES().to(args.device) for _ in range(args.cluster_n)] # Distribution means
    args.h = sqrt(args.n)
    args.c_cov = 1 / (3 * sqrt(args.n) + 5)
    args.c = 2 / (args.n + 7)
    args.p = [[torch.zeros_like(v) for k, v in synced_model[i].es_params()] for i in range(args.cluster_n)]
    args.history_theta = [[] for _ in range(args.cluster_n)]
    # Gradients will not be calculated
    for i in range(args.cluster_n):
        for param in synced_model[i].parameters():
            param.requires_grad = False
    # Restore models
    if args.restore:
        try:
            state_dict = torch.load(args.restore)
            synced_model[0].load_state_dict(state_dict)
            print(f'Restoring model from the specified file: {args.restore}')
        except:
            print('Restoring model from the provided file, but meet failure.')
    else:
        print('Starting a new training session.')
    # Test or train models
    if args.test:
        render_env(args, synced_model[0], args.test_case)
    else:
        train_loop(args, synced_model)