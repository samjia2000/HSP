#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import torch
import yaml
import numpy as np
from argparse import Namespace
from pathlib import Path

from hsp.config import get_config

from hsp.envs.overcooked.Overcooked_Env import Overcooked, OvercookedEnv
from hsp.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from hsp.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ChooseSubprocVecEnv, ChooseDummyVecEnv
from hsp.envs.wrappers.env_policy import PartialPolicyEnv

def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir)
                else:
                    env = Overcooked_new(all_args, run_dir)
                env = PartialPolicyEnv(all_args, env)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir)
                else:
                    env = Overcooked_new(all_args, run_dir)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ChooseSubprocVecEnv([get_env_fn(0)])
    else:
        return ChooseDummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--layout_name", type=str, default='cramped_room', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")
    parser.add_argument("--use_hsp", default=False, action='store_true')   
    parser.add_argument("--random_index", default=False, action='store_true')
    parser.add_argument("--use_agent_policy_id", default=False, action='store_true', help="Add policy id into share obs, default False")
    parser.add_argument("--predict_other_shaped_info", default=False, action='store_true', help="Predict other agent's shaped info within a short horizon, default False")
    parser.add_argument("--predict_shaped_info_horizon", default=50, type=int, help="Horizon for shaped info target, default 50")
    parser.add_argument("--predict_shaped_info_event_count", default=10, type=int, help="Event count for shaped info target, default 10")
    parser.add_argument("--shaped_info_coef", default=0.5, type=float)
    parser.add_argument("--policy_group_normalization", default=False, action="store_true")
    parser.add_argument("--use_advantage_prioritized_sampling", default=False, action='store_true')
    parser.add_argument("--uniform_preference", default=False, action='store_true')
    parser.add_argument("--uniform_sampling_repeat", default=0, type=int)
    parser.add_argument("--use_task_v_out", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float, help="Probability to use a random start state, default 0.")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    # population
    parser.add_argument("--population_yaml_path", type=str, help="Path to yaml file that stores the population info.")

    # mep
    parser.add_argument("--stage", type=int, default=1 ,help="Stages of MEP training. 1 for Maximum-Entropy PBT. 2 for FCP-like training.")
    parser.add_argument("--mep_use_prioritized_sampling", default=False, action='store_true', help="Use prioritized sampling in MEP stage 2.")
    parser.add_argument("--mep_prioritized_alpha", type=float, default=3.0, help="Alpha used in softing prioritized sampling probability.")
    parser.add_argument("--mep_entropy_alpha", type=float, default=0.01, help="Weight for population entropy reward. MEP uses 0.01 in general except 0.04 for Forced Coordination")
    # population
    parser.add_argument("--population_size", type=int, default=5, help="Population size involved in training.")
    parser.add_argument("--adaptive_agent_name", type=str, required=True, help="Name of training policy at Stage 2.")
    
    # train and eval batching
    parser.add_argument("--train_env_batch", type=int, default=1, help="Number of parallel threads a policy holds")
    parser.add_argument("--eval_env_batch", type=int, default=1, help="Number of parallel threads a policy holds")

    # fixed policy actions inside env threads
    parser.add_argument("--use_policy_in_env", default=True, action="store_false", help="Use loaded policy to move in env threads.")

    # eval with fixed policy
    parser.add_argument("--eval_policy", default="", type=str)

    parser.add_argument("--use_detailed_rew_shaping", default=False, action="store_true")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert all_args.algorithm_name in ["mep", "adaptive"]

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
                         reinit=True,
                         tags=all_args.wandb_tags)
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
    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
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

    #  override policy config
    population_config = yaml.load(open(all_args.population_yaml_path), yaml.Loader)
    override_policy_config = {}
    agent_name = all_args.adaptive_agent_name
    override_policy_config[agent_name] = (Namespace(use_agent_policy_id=all_args.use_agent_policy_id, 
                                                    predict_other_shaped_info=all_args.predict_other_shaped_info,
                                                    predict_shaped_info_horizon=all_args.predict_shaped_info_horizon,
                                                    predict_shaped_info_event_count=all_args.predict_shaped_info_event_count,
                                                    shaped_info_coef=all_args.shaped_info_coef,
                                                    policy_group_normalization=all_args.policy_group_normalization,
                                                    num_v_out=all_args.num_v_out,
                                                    use_task_v_out=all_args.use_task_v_out,
                                                    use_policy_vhead=all_args.use_policy_vhead), 
                                                    *runner.policy_config[1:])
    for policy_name in population_config:
        if policy_name != agent_name:
            override_policy_config[policy_name] = (None, None, runner.policy_config[2], None) # only override share_obs_space

    runner.policy.load_population(all_args.population_yaml_path, evaluation=False, override_policy_config=override_policy_config)
    runner.trainer.init_population()

    runner.train_mep()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])