from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from torch.nn.functional import pairwise_distance
from torch.autograd import Variable
from copy import deepcopy
from model import ES
from math import log, sqrt, exp
import matplotlib.pyplot as plt
import queue
from env import Bay_Env
import time
import sys

averageReward = []
maxReward = []
minReward = []
episodeCounter = []
really_reward_list = []


def update_progress_bar(progress):
    bar_length = 40
    block = int(round(bar_length * progress))
    progress_str = "[" + "=" * block + "-" * (bar_length - block) + "]"
    sys.stdout.write("\r" + progress_str + " {0:.1%}".format(progress))
    sys.stdout.flush()


# KDE calculation
def kernel_density_estimation(args, target_model, reference_models, h):
    target_params = torch.cat([v.flatten() for k, v in target_model.es_params()])
    reference_params = [torch.cat([v.flatten() for k, v in model.es_params()]) for model in reference_models]
    reference_params = torch.stack(reference_params)
    distances = pairwise_distance(target_params.unsqueeze(0), reference_params)
    kde = torch.sum(torch.exp(-0.5 * (distances / h) ** 2))
    return kde / args.h


def do_rollouts(args, model, random_seeds, return_queue, test=False, case=None):
    if case is None:
        env = Bay_Env(args.bay_case_file)
    else:
        env = Bay_Env(case)
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = torch.from_numpy(state)
    this_model_return = 0
    this_model_num_frames = 0
    with torch.no_grad():
        for step in range(args.max_episode_length):
            state = state.float()
            state = state.unsqueeze(0)
            action = model(Variable(state).to(args.device))
            state, reward, done, _, _ = env.step(action)
            this_model_return += reward
            this_model_num_frames += 1
            if done:
                break
            state = torch.from_numpy(state)
    if test:
        return this_model_return
    else:
        return_queue.put((random_seeds, this_model_return, this_model_num_frames))


def perturb_model(args, model, new_model, random_seed, i):
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


def gradient_update(args, synced_model, returns, random_seeds,
                    num_eps, num_frames, unperturbed_results):
    # Count the number of subpopulation better than their distribution mean
    def unperturbed_rank(returns, unperturbed_results):
        nth_place = [0 for _ in range(args.cluster_n)]
        for i in range(args.cluster_n):
            for r in returns[i]:
                if r > unperturbed_results[i]:
                    nth_place[i] += 1
        rank_diag = (f'{nth_place} out of theta')
        return rank_diag, nth_place
    # Calculate the rank in each subpopulation
    sort_returns = [[] for _ in range(args.cluster_n)]
    rankings = [[0 for _ in range(args.lambda_)] for _ in range(args.cluster_n)]
    for i in range(args.cluster_n):
        sort_returns[i] = sorted([(r, num) for num, r in enumerate(returns[i])], key=lambda x: x[0])[::-1]
        for rank, j in enumerate(sort_returns[i]):
            rankings[i][j[1]] = rank

    rank_diag, rank = unperturbed_rank(returns, unperturbed_results)
    args.rank_d = rank
    # Test distribution means for visualization
    really_reward = [do_rollouts(args, synced_model[i], i, None, test=True, case=args.test_case) for i in
                     range(args.cluster_n)]

    if not args.silent:
        print(f'Iteration num: {args.episode_num}\n'
              f'Episode num: {num_eps}\n'
              f'Average reward: {np.mean(really_reward)}\n'
              f'Variance in rewards: {np.var(really_reward)}\n'
              f'Max reward on test case: {np.max(really_reward)}\n'
              f'Min reward on test case: {np.min(really_reward)}\n'
              f'Total population number: {args.cluster_n*args.lambda_}\n'
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

    np.save(args.folder_path + "/really_reward.npy", np.array(really_reward_list))
    np.save(args.folder_path + "/max_reward.npy", np.array(maxReward))

    pltMax, = plt.plot(range(len(episodeCounter)), maxReward, label='max')
    plt.ylabel('rewards')
    plt.xlabel('Iteration num')
    plt.legend(handles=[pltMax])
    plt.xticks(np.arange(0, len(episodeCounter), max(int(len(episodeCounter) // 10), 1)))

    fig1 = plt.gcf()

    plt.draw()
    fig1.savefig(args.folder_path + '/graph.png', dpi=100)

    # Upgrade new distribution means
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

    return synced_model

# Test model
def render_env(args, model, case=None):
    with torch.no_grad():
        if case is None:
            env = Bay_Env(args.bay_case_file)
        else:
            env = Bay_Env(case)
        all_returns = []
        all_num_frames = []
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = torch.from_numpy(state)
        this_model_return = 0
        this_model_num_frames = 0
        with torch.no_grad():
            for step in range(args.max_episode_length):
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
            print('Reward: %f' % this_model_return)


def generate_seeds_and_models(args, synced_model, new_model, i):
    np.random.seed()
    random_seed = np.random.randint(2 ** 31 -1)
    model = perturb_model(args, synced_model, new_model, random_seed, i)
    return random_seed, model.to(args.device)


# Upgrade algorithm parameters
def upgrade_args(args, model_parameters, last_model_parameters, results, last_results):
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
        q[i] = 2 * min(args.rank_d[i], args.miu) / args.miu - 1

    for i in range(args.cluster_n):
        args.s[i] = (1 - args.c_s) * args.s[i] + args.c_s * (q[i] - args.q_)
        args.sigma[i] = args.sigma[i] * exp(args.s[i] / args.d_sig)

    for i in range(args.cluster_n):
        if args.sigma[i] < 1e-6:
            args.need_restart[i] = True
            print(f"Restart {i} population for convergence")

    for i in range(args.cluster_n):
        if args.sigma[i] > 1:
            args.sigma[i] = 0.05
            print(f"Restart {i} population for sigma too large")


# Train models
def train_loop(args, synced_model):
    print("Num params in network %d" % synced_model[0].count_parameters())
    all_models = [[ES().to(args.device) for _ in range(args.lambda_)] for _ in range(args.cluster_n)]
    num_eps = 0
    total_num_frames = 0
    last_results = [[] for _ in range(args.cluster_n)]
    for loop in range(args.train_loop):
        for case in range(args.train_case_num):
            args.bay_case_file = args.train_folder + f"/{case}.txt"
            args.episode_num = case + loop * args.train_case_num
            return_queue = [queue.Queue() for _ in range(args.cluster_n)]
            all_seeds = [[] for _ in range(args.cluster_n)]
            last_model_parameters = [[deepcopy(v) for k, v in synced_model[i].es_params()] for i in
                                     range(args.cluster_n)]
            # Generate sub-populations from distribution means
            for i in range(int(args.cluster_n)):
                num_p = 0
                while num_p < int(args.lambda_):
                    random_seed, model = generate_seeds_and_models(args, synced_model[i], all_models[i][num_p], i)
                    flag = True
                    if args.cluster_n > 1:
                        rol = kernel_density_estimation(args, model, synced_model[:i] + synced_model[i + 1:] + [y for x in args.history_theta[:i] for y in x] + [y for x in args.history_theta[i + 1:] for y in x], args.h)
                        if rol > args.r1:
                            flag = False
                    if flag:
                        num_p += 1
                        all_seeds[i].append(random_seed)
                        all_models[i].append(model)
                assert len(all_seeds) == len(all_models)
            # Test on training cases
            t = time.time()
            test_num = 0
            for i in range(int(args.cluster_n)):
                for j in range(args.lambda_):
                    perturbed_model = all_models[i][j]
                    seed = all_seeds[i][j]
                    do_rollouts(args, perturbed_model, seed, return_queue[i], test=False)
                    test_num += 1
                    update_progress_bar(test_num / (args.cluster_n * args.lambda_))
                do_rollouts(args, synced_model[i], 'dummy_seed', return_queue[i], test=False)
            print(f"\nThe {args.episode_num}th iteration is finished:", time.time() - t, "s")
            # Separate reward
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
            # Upgrade partial parameters
            total_num_frames += sum([sum(k) for k in num_frames])
            num_eps += sum([len(k) for k in results])
            args.miu = int(
                (args.episode_num / (args.train_loop * args.train_case_num)) * (args.lambda_ - args.miu_0) + args.miu_0)
            args.sum_miu = sum([log(j) for j in range(1, args.miu + 1)])
            args.w = [(log((args.miu + 1) / i)) / (args.miu * log(args.miu + 1) - args.sum_miu) for i in
                      range(1, args.miu + 1)]
            args.miu_eff = 1 / (sum([w * w for w in args.w]))
            assert args.miu <= args.lambda_
            # Upgrade distribution means
            synced_model = gradient_update(args, synced_model, results, seeds,
                                           num_eps, total_num_frames,
                                           unperturbed_results)
            model_parameters = [[deepcopy(v) for k, v in synced_model[i].es_params()] for i in range(args.cluster_n)]
            upgrade_args(args, model_parameters, last_model_parameters, results, last_results)
            last_results = deepcopy(results)
            # Calculate the ranks of distribution means on the current case
            rank = []
            main_return_queue = queue.Queue()
            for i in range(args.cluster_n):
                do_rollouts(args, synced_model[i], i, main_return_queue, test=False)
            for _ in range(args.cluster_n):
                seed, reward, frame = main_return_queue.get()
                rank.append((reward, seed))
            rank.sort(key=lambda x: x[0], reverse=True)
            mean_reward = sum([reward for reward, i in rank]) / args.cluster_n
            # Population exclusion according to rank
            if args.cluster_n > 1:
                for reward, i in rank:
                    rou = kernel_density_estimation(args, synced_model[i],
                                                    synced_model[:i] + synced_model[i + 1:] +  args.history_theta[i],
                                                    args.h)
                    if rou > args.r2 or args.need_restart[i]:
                        if reward >= mean_reward or i <= (args.cluster_n // 2):
                            print(f"The {i}th population's parameters is restarted for KDE:{rou}")
                            args.need_restart[i] = False
                            args.sigma[i] = 0.05
                            args.p[i] = [torch.zeros_like(v) for v in model_parameters[i]]
                            args.s[i] = 0
                            last_results[i] = []
                        else:
                            args.history_theta[i] = []
                            print(f"The {i}th population is deleted for KDE:{rou}")
                            # Find new mean that is in the low density region
                            while True:
                                synced_model[i].init_weight()
                                rou = kernel_density_estimation(args, synced_model[i],
                                                                synced_model[:i] + synced_model[
                                                                                   i + 1:] + args.history_theta[i],
                                                                args.h)
                                if rou <= args.r2:
                                    break
                            args.need_restart[i] = False
                            args.sigma[i] = 0.05
                            args.p[i] = [torch.zeros_like(v) for v in model_parameters[i]]
                            args.s[i] = 0
                            last_results[i] = []
            # Save distribution means and history trajectory
            for i in range(args.cluster_n):
                torch.save(synced_model[i].state_dict(),
                           args.folder_path + rf'/synced_model_{i}_{args.para_number}.pth')
                args.history_theta[i].append(deepcopy(synced_model[i]))
