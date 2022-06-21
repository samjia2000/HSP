#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
import yaml
import pickle
from pathlib import Path

import torch

from hsp.config import get_config

from hsp.envs.overcooked.Overcooked_Env import Overcooked
from hsp.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from hsp.envs.env_wrappers import  ChooseSubprocVecEnv, ChooseDummyVecEnv

def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    ENV = Overcooked
                else:
                    ENV = Overcooked_new
                env = ENV(all_args, run_dir, featurize_type=("bc", "bc"), rank=rank)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--layout_name", type=str, default='cramped_room', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")  
    parser.add_argument("--use_hsp", default=False, action='store_true')   
    parser.add_argument("--random_index", default=False, action='store_true')
    parser.add_argument("--use_agent_policy_id", default=False, action='store_true', help="Add policy id into share obs, default False")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    parser.add_argument("--use_detailed_rew_shaping", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float)
    parser.add_argument("--store_traj", default=False, action='store_true')
    # population
    parser.add_argument("--population_yaml_path", type=str, help="Path to yaml file that stores the population info.")
    
    # overcooked evaluation
    parser.add_argument("--agent0_policy_name", type=str, help="policy name of agent 0")
    parser.add_argument("--agent1_policy_name", type=str, help="policy name of agent 1")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert all_args.algorithm_name == "population"

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.layout_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_eval_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from hsp.runner.shared.overcooked_runner import OvercookedRunner as Runner
    else:
        from hsp.runner.separated.overcooked_runner import MPERunner as Runner

    runner = Runner(config)

    # load population
    print("population_yaml_path: ",all_args.population_yaml_path)
    featurize_type = runner.policy.load_population(all_args.population_yaml_path, evaluation=True)

    # configure mapping from (env_id, agent_id) to policy_name
    map_ea2p = dict()
    for e in range(all_args.n_eval_rollout_threads):
        map_ea2p[(e, 0)] = all_args.agent0_policy_name
        map_ea2p[(e, 1)] = all_args.agent1_policy_name
    runner.policy.set_map_ea2p(map_ea2p)

    # set featurize_type of eval threaded env
    agent0_featurize_type = featurize_type.get(all_args.agent0_policy_name, "ppo")
    agent1_featurize_type = featurize_type.get(all_args.agent1_policy_name, "ppo")
    eval_envs.reset_featurize_type([(agent0_featurize_type, agent1_featurize_type) for _ in range(all_args.n_eval_rollout_threads)])

    runner.evaluate_with_multi_policy()

    if envs is not None:
        # post process
        envs.close()
    if eval_envs is not None:
        # post process
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])