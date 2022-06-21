    
import time
from typing import Any, Dict
import wandb
import os
import socket
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
import pickle

from hsp.utils.separated_buffer import SeparatedReplayBuffer
from hsp.utils.util import update_linear_schedule

import psutil
import slackweb
import socket
webhook_url = " https://hooks.slack.com/services/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"

def _t2n(x):
    return x.detach().cpu().numpy()

def make_trainer_policy_cls(algorithm_name, use_single_network=False):
    if "mappo" in algorithm_name:
        if use_single_network:
            from hsp.algorithms.r_mappo_single.r_mappo_single import R_MAPPO as TrainAlgo
            from hsp.algorithms.r_mappo_single.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        else:
            from hsp.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from hsp.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
    elif "mappg" in algorithm_name:
        if use_single_network:
            from hsp.algorithms.r_mappg_single.r_mappg_single import R_MAPPG as TrainAlgo
            from hsp.algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
        else:
            from hsp.algorithms.r_mappg.r_mappg import R_MAPPG as TrainAlgo
            from hsp.algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
    elif "overcooked_bc" == algorithm_name:
        from hsp.envs.overcooked.human_aware_rl.imitation.behavior_cloning import BehaviorCloneTrainer as TrainAlgo
        from hsp.envs.overcooked.human_aware_rl.imitation.behavior_cloning import BehaviorClonePolicy as Policy
    elif "ft" in algorithm_name:
        print("use frontier-based algorithm")
    elif "population" == algorithm_name:
        from hsp.algorithms.population.trainer_pool import TrainerPool as TrainAlgo
        from hsp.algorithms.population.policy_pool import PolicyPool as Policy
    elif algorithm_name == "mep":
        from hsp.algorithms.population.mep import MEP_Trainer as TrainAlgo
        from hsp.algorithms.population.policy_pool import PolicyPool as Policy
    elif algorithm_name == "traj":
        from hsp.algorithms.population.traj import Traj_Trainer as TrainAlgo
        from hsp.algorithms.population.policy_pool import PolicyPool as Policy
    else:
        raise NotImplementedError
    return TrainAlgo, Policy

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.use_single_network = self.all_args.use_single_network
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.run_dir = self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        TrainAlgo, Policy = make_trainer_policy_cls(self.algorithm_name, use_single_network=self.use_single_network)

        # dump policy config to allow loading population in yaml form
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        if hasattr(self.all_args, "use_agent_policy_id") and self.all_args.use_agent_policy_id:
            share_observation_space = share_observation_space[:-1] + [[self.all_args.num_agents, 1, False, config['policy_pool_size']]] +  [share_observation_space[-1]]
            share_observation_space[0] += self.all_args.num_agents
        self.policy_config = (self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0])
        policy_config_path = os.path.join(self.run_dir, 'policy_config.pkl')
        pickle.dump(self.policy_config, open(policy_config_path, "wb"))
        print(f"Pickle dump policy config at {policy_config_path}")

        if self.algorithm_name in ["population", "mep", "traj"]:
            # policy network
            self.policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                device = self.device)

            if self.model_dir is not None:
                self.restore()

            # algorithm
            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        else:
            self.policy = []
            for agent_id in range(self.num_agents):
                #print(len(self.envs.share_observation_space))
                #print(len(self.envs.observation_space))
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
                # policy network
                po = Policy(self.all_args,
                            self.envs.observation_space[agent_id],
                            share_observation_space,
                            self.envs.action_space[agent_id],
                            device = self.device)
                self.policy.append(po)

            if self.model_dir is not None:
                self.restore()

            self.trainer = []
            self.buffer = []
            for agent_id in range(self.num_agents):
                # algorithm
                tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
                # buffer
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
                #print("Base runner", agent_id, share_observation_space)
                bu = SeparatedReplayBuffer(self.all_args,
                                        self.envs.observation_space[agent_id],
                                        share_observation_space,
                                        self.envs.action_space[agent_id])
                self.buffer.append(bu)
                self.trainer.append(tr)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()
        self.log_system()
        return train_infos

    def save(self, steps=None):
        postfix = f"_{steps}.pt" if steps else ".pt"
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + postfix)
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + postfix)
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + postfix)

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                if not self.use_render:
                    policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                    self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log(self, infos: Dict[str, Any], step):
        if self.use_wandb:
            wandb.log(infos, step=step)
        else:
            [self.writter.log(k, v, step) for k, v in infos.items()]

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def log_system(self):
        # RRAM
        mem = psutil.virtual_memory()
        total_mem = float(mem.total) / 1024 / 1024 / 1024
        used_mem = float(mem.used) / 1024 / 1024 / 1024
        if used_mem/total_mem > 0.95:
            slack = slackweb.Slack(url=webhook_url)
            host_name = socket.gethostname()
            slack.notify(text="Host {}: occupied memory is *{:.2f}*%!".format(host_name, used_mem/total_mem*100))