from __future__ import absolute_import, division, print_function
import os
import argparse
import torch
from math import log,sqrt
from model import ES
from train import train_loop, render_env
from env import Bay_Env

parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--env-name', default='BACAP',
                    metavar='ENV', help='environment')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-decay', type=float, default=1, metavar='LRD',
                    help='learning rate decay')
parser.add_argument('--sigma', type=float, default=0.05, metavar='SD',
                    help='noise standard deviation')
parser.add_argument('--useAdam', action='store_true',
                    help='bool to determine if to use adam optimizer')
parser.add_argument('--n', type=int, default=40, metavar='N',
                    help='batch size, must be even')
parser.add_argument('--max-episode-length', type=int, default=100000,
                    metavar='MEL', help='maximum length of an episode')
parser.add_argument('--max-gradient-updates', type=int, default=100000,
                    metavar='MGU', help='maximum number of updates')
parser.add_argument('--restore', default='', metavar='RES',
                    help='checkpoint from which to restore')
parser.add_argument('--small-net', action='store_true',
                    help='Use simple MLP on CartPole')
parser.add_argument('--variable-ep-len', action='store_true',
                    help="Change max episode length during training")
parser.add_argument('--silent', action='store_true',
                    help='Silence print statements during training')
parser.add_argument('--test', action='store_true',
                    help='Just render the env, no training')



if __name__ == '__main__':
    for i in range(1000):
        folder_name = str(i)
        folder_path = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(folder_path)
            break

    args = parser.parse_args()
    args.folder_path = folder_path
    args.device = torch.device("cpu")
    args.small_net = False
    args.restore = False
    args.test = False
    args.max_gradient_updates = 10000
    args.train_loop = 1
    args.max_episode_length = 1000
    args.env_name = "Bay"
    args.train_flood = r"./case_code"
    args.test_case = r"./case_code/test.txt"
    args.train_case_num = 200
    args.useAdam = False
    args.use_restart = False
    args.restart_time = 0
    args.max_restart = 10
    args.store_para = dict()
    args.store_max_num = 3
    args.store_baseline = -300000
    args.history_theta = []
    args.use_best = False

    args.n = 100
    args.r1 = 0.5
    args.r2 = 0.2
    args.h = sqrt(args.n)
    args.cluster_n = 5
    args.lambda_ = 80
    args.need_re = [False for _ in range(args.cluster_n)]
    args.history_results = [[] for _ in range(args.cluster_n)]
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
    args.theta = [0 for _ in range(args.cluster_n)]
    args.sigma = [0.05 for _ in range(args.cluster_n)]
    args.p = [[] for _ in range(args.cluster_n)]
    args.s = [0 for _ in range(args.cluster_n)]
    args.prev_reward = [[] for _ in range(args.cluster_n)]
    args.lr = 1
    args.lr_decay = 1
    args.para_number = 800
    env = Bay_Env(args.test_case)
    chkpt_dir = 'checkpoints/%s/' % args.env_name
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    synced_model = [ES().to(args.device) for _ in range(args.cluster_n)]
    args.para_number = synced_model[0].count_parameters()
    args.n = args.para_number
    args.h = sqrt(args.n)
    args.n = args.max_gradient_updates
    args.c_cov = 1 / (3 * sqrt(args.n) + 5)
    args.c = 2 / (args.n + 7)
    args.p = [[torch.zeros_like(v) for k, v in synced_model[i].es_params()] for i in range(args.cluster_n)]
    for i in range(args.cluster_n):
        for param in synced_model[i].parameters():
            param.requires_grad = False
    if args.restore:
        state_dict = torch.load(args.restore)
        synced_model[0].load_state_dict(state_dict)
    if args.test:
        render_env(args, synced_model, env)
    else:
        train_loop(args, synced_model, env, chkpt_dir)
