from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from torch.nn.functional import pairwise_distance
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from model import ES
from math import log, sqrt, exp
import matplotlib.pyplot as plt
import queue
from env import Bay_Env
import time
import sys


def update_progress_bar(progress):
    bar_length = 40
    block = int(round(bar_length * progress))
    progress_str = "[" + "=" * block + "-" * (bar_length - block) + "]"
    sys.stdout.write("\r" + progress_str + " {0:.1%}".format(progress))
    sys.stdout.flush()


def kernel_density_estimation(args, target_model, reference_models, h):
    target_params = torch.cat([v.flatten() for k, v in target_model.es_params()])

    reference_params = [torch.cat([v.flatten() for k, v in model.es_params()]) for model in reference_models]

    reference_params = torch.stack(reference_params)

    distances = pairwise_distance(target_params.unsqueeze(0), reference_params)

    kde = torch.sum(torch.exp(-0.5 * (distances / h) ** 2))

    return kde / args.h


def calculate_model_distance(model1, model2):
    params1 = dict(model1.es_params())
    params2 = dict(model2.es_params())

    assert len(params1) == len(params2)

    distance = 0.0
    for k in params1.keys():
        assert k in params2
        p1 = params1[k]
        p2 = params2[k]
        distance += ((p1 - p2) ** 2).sum().item()

    return distance


def average_model_weights(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        param1.data = (param1.data + param2.data) / 2.0


def do_rollouts(args, models, random_seeds, return_queue, test=False,case = None):
    if case is None:
        env = Bay_Env(args.bay_case_file)
    else:
        env = Bay_Env(case)
    assert len(models) == 1
    all_returns = []
    all_num_frames = []
    for model in models:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = torch.from_numpy(state)
        this_model_return = 0
        this_model_num_frames = 0
        with torch.no_grad():
            for step in range(args.max_episode_length):
                if args.small_net:
                    state = state.float()
                    state = state.view(1, env.observation_space.shape[0])
                    logit = model(Variable(state).to(args.device))
                    prob = F.softmax(logit, dim=-1)
                    action = prob.max(1)[1].cpu().data.numpy()
                    action = action[0]
                else:
                    state = state.float()
                    state = state.unsqueeze(0)
                    action = model(Variable(state).to(args.device))
                state, reward, done, _, _ = env.step(action)
                this_model_return += reward
                this_model_num_frames += 1
                if done:
                    break
                state = torch.from_numpy(state)
            all_returns.append(this_model_return)
            all_num_frames.append(this_model_num_frames)
    if test:
        return all_returns
    assert len(all_returns) == 1

    return_queue.put((random_seeds[0], all_returns[0], all_num_frames[0]))


def perturb_model(args, model, new_model, random_seed, env, i):
    new_model.load_state_dict(model.state_dict())
    torch.manual_seed(random_seed)
    index = 0
    for k, v in new_model.es_params():
        z = torch.randn(v.size(), device=args.device)
        r = torch.randn(v.size(), device=args.device)
        perturb = args.sigma[i] * (sqrt(1 - args.c_cov) * z) + sqrt(args.c_cov) * torch.mul(args.p[i][index], r)
        v += perturb.float().to(args.device)
        index += 1
    return new_model


optimConfig = []
averageReward = []
maxReward = []
minReward = []
episodeCounter = []
really_reward_list = []


def gradient_update(args, synced_model, returns, random_seeds,
                    num_eps, num_frames, chkpt_dir, unperturbed_results):
    def unperturbed_rank(returns, unperturbed_results):
        nth_place = [0 for _ in range(args.cluster_n)]
        for i in range(args.cluster_n):
            for r in returns[i]:
                if r > unperturbed_results[i]:
                    nth_place[i] += 1
        rank_diag = (f'{nth_place} out of theta')
        return rank_diag, nth_place

    batch_size = len(returns[0])
    assert batch_size == args.lambda_
    assert len(random_seeds[0]) == batch_size
    sort_returns = [[] for _ in range(args.cluster_n)]
    rankings = [[0 for _ in range(batch_size)] for _ in range(args.cluster_n)]
    for i in range(args.cluster_n):
        sort_returns[i] = sorted([(r, num) for num, r in enumerate(returns[i])], key=lambda x: x[0])[::-1]
        for rank, j in enumerate(sort_returns[i]):
            rankings[i][j[1]] = rank

    rank_diag, rank = unperturbed_rank(returns, unperturbed_results)
    args.rank_d = rank

    really_reward = [do_rollouts(args, [synced_model[i]], [i], [], test=True,case=args.test_case) for i in range(args.cluster_n)]


    if not args.silent:
        print(f'Iteration num: {args.episode_num}\n'
              f'Episode num: {num_eps}\n'
              f'Average reward: {np.mean(really_reward)}\n'
              f'Variance in rewards: {np.var(really_reward)}\n'
              f'Max reward: {np.max(really_reward)}\n'
              f'Min reward: {np.min(really_reward)}\n'
              f'Batch size: {batch_size}\n'
              f'Max episode length: {args.max_episode_length}\n'
              f'Sigma: {args.sigma}\n'
              f's: {args.s}\n'
              f'Learning rate: {args.lr}\n'
              f'Total num frames seen: {num_frames}\n'
              f'Test reward: {really_reward}\n'
              f'Unperturbed reward: {unperturbed_results}\n'
              f'Unperturbed rank: {rank_diag}\n')


    averageReward.append(np.mean(really_reward))
    episodeCounter.append(num_eps)
    maxReward.append(np.max(really_reward))
    minReward.append(np.min(really_reward))
    really_reward_list.append(really_reward)

    np.save(args.folder_path+"/really_reward.npy",np.array(really_reward_list))
    np.save( args.folder_path+"/max_reward.npy",np.array(maxReward))

    pltMax, = plt.plot(range(len(episodeCounter)), maxReward, label='max')
    plt.ylabel('rewards')
    plt.xlabel('Iteration num')
    plt.legend(handles=[pltMax])
    plt.xticks(np.arange(0, len(episodeCounter), max(int(len(episodeCounter) // 10), 1)))

    fig1 = plt.gcf()

    plt.draw()
    fig1.savefig(args.folder_path+'/graph.png', dpi=100)

    if True:
        for i in range(args.cluster_n):
            for j in range(args.lambda_):
                if rankings[i][j] >= args.miu: 
                    continue
                torch.manual_seed(random_seeds[i][j])
                index = 0
                for k, v in synced_model[i].es_params():
                    z = torch.randn(v.size(), device=args.device)
                    r = torch.randn(v.size(), device=args.device)
                    perturb = args.w[rankings[i][j]] * args.sigma[i] * (
                            sqrt(1 - args.c_cov) * z + sqrt(args.c_cov) * torch.mul(args.p[i][index], r))
                    v += (args.lr * perturb).float()
                    index += 1
        args.lr *= args.lr_decay

    return synced_model


def render_env(args, model, env):
    with torch.no_grad():
        while True:
            state = env.reset()
            if len(state) > 1:
                state = state[0]
            state = torch.from_numpy(state)
            this_model_return = 0
            if not args.small_net:
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            done = False
            while not done:
                if args.small_net:
                    state = state.float()
                    state = state.view(1, env.observation_space.shape[0])
                    logit = model(Variable(state, volatile=True))
                else:
                    logit, (hx, cx) = model(
                        (Variable(state.unsqueeze(0), volatile=True),
                         (hx, cx)))
                prob = F.softmax(logit, dim=-1)
                action = prob.max(1)[1].cpu().data.numpy()
                state, reward, done, _, _ = env.step(action[0])
                env.render()
                this_model_return += reward
                state = torch.from_numpy(state)
            print('Reward: %f' % this_model_return)


def generate_seeds_and_models(args, synced_model, new_model, env, i):
    np.random.seed()
    random_seed = np.random.randint(2 ** 30)
    model = perturb_model(args, synced_model, new_model, random_seed, env, i)
    return random_seed, model.to(args.device)


def upgrade_args(args, model_parameters, last_model_parameters, results, last_results):
    args.prev_reward += [item for sublist in results for item in sublist]
    if len(args.prev_reward) > args.n:
        args.prev_reward = args.prev_reward[-args.n:]
    p = [[] for i in range(args.cluster_n)]
    for i in range(args.cluster_n):
        index = 0
        for v1, v2 in zip(last_model_parameters[i], model_parameters[i]):
            p[i].append(
                (1 - args.c) * args.p[i][index] + sqrt(args.c * (2 - args.c) * args.miu_eff) * (v2 - v1) / args.sigma[
                    i])
            index += 1
    args.p = p

    q = [0 for _ in range(args.cluster_n)]
    for i in range(args.cluster_n):
        q[i] = 2*min(args.rank_d[i],args.miu)/args.miu -1

    for i in range(args.cluster_n):
        args.s[i] = (1 - args.c_s) * args.s[i] + args.c_s * (q[i] - args.q_)
        args.sigma[i] = args.sigma[i] * exp(args.s[i] / args.d_sig)

    for i in range(args.cluster_n):
        if args.sigma[i] < 1e-6:
            args.need_re[i] = True
            print(f"Restart {i} population for convergence")

    for i in range(args.cluster_n):
        if args.sigma[i] > 1:
            args.sigma[i] = 0.05
            print(f"Restart {i} population for sigma too large")


def train_loop(args, synced_model, env, chkpt_dir):
    print("Num params in network %d" % synced_model[0].count_parameters())
    all_models = [[ES().to(args.device) for _ in range(args.lambda_)] for _ in range(args.cluster_n)]
    num_eps = 0
    total_num_frames = 0
    last_results = [[] for _ in range(args.cluster_n)]
    for loop in range(args.train_loop):
        for case in range(args.train_case_num):
            args.bay_case_file = args.train_flood+f"/{case}.txt"
            args.episode_num = case + loop * args.train_case_num
            return_queue = [queue.Queue() for _ in range(args.cluster_n)]
            all_seeds = [[] for _ in range(args.cluster_n)]
            last_model_parameters = [[deepcopy(v) for k, v in synced_model[i].es_params()] for i in
                                     range(args.cluster_n)]
            for i in range(int(args.cluster_n)):
                num_p = 0
                while num_p < int(args.lambda_):
                    random_seed, model = generate_seeds_and_models(args, synced_model[i], all_models[i][num_p], env, i)
                    flag = True
                    if args.cluster_n > 1:
                        rol = kernel_density_estimation(args, model, synced_model[:i] + synced_model[i + 1:], args.h)
                        if rol > args.r1:
                            flag = False
                    if flag:

                        num_p += 1
                        all_seeds[i].append(random_seed)
                        all_models[i].append(model)
                assert len(all_seeds) == len(all_models)

            t = time.time()
            test_num = 0
            for i in range(int(args.cluster_n)):
                for j in range(args.lambda_):
                    perturbed_model = all_models[i][j]
                    seed = all_seeds[i][j]
                    do_rollouts(args, [perturbed_model], [seed], return_queue[i], test=False)
                    test_num += 1
                    update_progress_bar(test_num / (args.cluster_n * args.lambda_))
                do_rollouts(args, [synced_model[i]], ['dummy_seed'], return_queue[i], test=False)
            print(f"\n{args.episode_num}结束:", time.time() - t, "s")

            unperturbed_results = []
            results, seeds, num_frames = [[] for i in range(args.cluster_n)], [[] for i in range(args.cluster_n)], [[] for i in range(args.cluster_n)]
            for i in range(args.cluster_n):
                for _ in range(args.lambda_ + 1):
                    seed, reward, frame = return_queue[i].get()
                    if seed == 'dummy_seed':
                        unperturbed_results.append(reward)
                        continue
                    results[i].append(reward)
                    seeds[i].append(seed)
                    num_frames[i].append(frame)
            assert len(unperturbed_results) == args.cluster_n

            total_num_frames += sum([sum(k) for k in num_frames])
            num_eps += sum([len(k) for k in results])

            args.miu = int((args.episode_num / (args.train_loop * args.train_case_num)) * (args.lambda_ - args.miu_0) + args.miu_0)
            args.sum_miu = sum([log(j) for j in range(1, args.miu + 1)])
            args.w = [(log((args.miu + 1) / i)) / (args.miu * log(args.miu + 1) - args.sum_miu) for i in
                      range(1, args.miu + 1)]
            args.miu_eff = 1 / (sum([w * w for w in args.w]))
            assert args.miu <= args.lambda_

            synced_model = gradient_update(args, synced_model, results, seeds,
                                           num_eps, total_num_frames,
                                           chkpt_dir, unperturbed_results)

            model_parameters = [[deepcopy(v) for k, v in synced_model[i].es_params()] for i in range(args.cluster_n)]
            if args.variable_ep_len:
                args.max_episode_length = int(2 * sum(num_frames) / len(num_frames))

            upgrade_args(args, model_parameters, last_model_parameters, results, last_results)

            last_results = deepcopy(results)

            rank = []
            main_return_queue = queue.Queue()
            for i in range(args.cluster_n):
                do_rollouts(args, [synced_model[i]], [i], main_return_queue, test=False)
            for _ in range(args.cluster_n):
                seed, reward, frame = main_return_queue.get()
                rank.append((reward, seed))
            rank.sort(key=lambda x: x[0], reverse=True)
            mean_reward = sum([reward for reward, i in rank])/args.cluster_n
            if args.cluster_n > 1:
                for reward, i in rank:
                    rou = kernel_density_estimation(args, synced_model[i],
                                                    synced_model[:i] + synced_model[i + 1:] + args.history_theta, args.h)
                    if rou > args.r2 or args.need_re[i]:
                        if reward >= mean_reward or i <= (args.cluster_n//2):
                            print(f"第{i}个种群被重启参数:{rou}")
                            args.need_re[i] = False
                            args.sigma[i] = 0.05
                            args.p[i] = [torch.zeros_like(v) for v in model_parameters[i]]
                            args.s[i] = 0
                            last_results[i] = []

                        else:
                            print(f"第{i}个种群被删除:{rou}")
                            args.history_theta.append(deepcopy(synced_model[i]))
                            while True:
                                synced_model[i].init_weight()
                                rou = kernel_density_estimation(args, synced_model[i],
                                                                synced_model[:i] + synced_model[i + 1:] + args.history_theta,
                                                                args.h)
                                if rou <= args.r2:
                                    break
                            args.need_re[i] = False
                            args.sigma[i] = 0.05
                            args.p[i] = [torch.zeros_like(v) for v in model_parameters[i]]
                            args.s[i] = 0
                            last_results[i] = []
                    for i in range(args.cluster_n):
                        torch.save(synced_model[i].state_dict(), args.folder_path+rf'/synced_model_{i}_{args.para_number}.pth')

