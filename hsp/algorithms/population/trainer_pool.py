import os
import copy
from typing import Dict
import numpy as np
import torch
from collections import defaultdict
from hsp.algorithms.population.policy_pool import PolicyPool
from hsp.runner.shared.base_runner import make_trainer_policy_cls
from hsp.utils.shared_buffer import SharedReplayBuffer
from hsp.algorithms.population.utils import _t2n

class TrainerPool:
    """TrainerPool maintains a pool of trainers, each trainer corresponding to one policy, both have the same name. 
    For policies that are not trained, use null trainer.
    By specifying mapping from (env_id, agent_id) to trainer_name, TrainerPool creates buffer for each policy.
    """
    def __init__(self, args, policy_pool: PolicyPool, device = torch.device("cpu")):
        self.all_args = args
        self.device = device
        self.policy_pool = policy_pool
        self.trainer_pool = {}
        self.trainer_total_num_steps = defaultdict(int)
        self.use_policy_in_env = dict(args._get_kwargs()).get('use_policy_in_env', False)
        self.__loaded_population = False
        self.__initialized = False

    def policy_config(self, trainer_name):
        return self.policy_pool.policy_config[trainer_name]

    def policy_type(self, trainer_name):
        if trainer_name.startswith("ppo") and trainer_name[-1] in "123":
            return eval(trainer_name[-1])
        elif trainer_name.startswith("policy"):
            # preference policy
            return 4
        else:
            raise RuntimeError(f"Cannot recognize policy type for {trainer_name}.")

    def policy_id(self, trainer_name):
        return int(self.policy_pool.policy_info[trainer_name][1]["id"] * self.policy_pool.num_policies - 1)

    def init_population(self):
        self.on_training = []
        self.best_r = defaultdict(float)
        for policy_name, policy, policy_config, policy_train in self.policy_pool.all_policies():
            # use the same name for trainer and policy
            trainer_name = policy_name
            trainer_cls, _ = make_trainer_policy_cls(policy_config[0].algorithm_name, use_single_network=policy_config[0].use_single_network)
            trainer = trainer_cls(policy_config[0], policy, device = self.device)
            self.trainer_pool[trainer_name] = trainer
            self.best_r[trainer_name] = -1e9
            if policy_train:
                self.on_training.append(trainer_name)
        
        # trans policies in policy pool to EvalPolicy
        self.policy_pool.trans_to_eval()

        # train info would update when a trainer performs training
        self.train_infos = {}
        self.train_infos.update({f"{trainer_name}-total_num_steps":0 for trainer_name in self.trainer_pool.keys()})

        self.__loaded_population = True

    def reset(self, map_ea2t, n_rollout_threads, num_agents, load_unused_to_cpu=False, **kwargs):
        assert self.__loaded_population
        self.map_ea2t = map_ea2t
        self.n_rollout_threads = n_rollout_threads
        self.num_agents = num_agents

        self.control_agent_count = defaultdict(int)
        self.control_agents = defaultdict(list)
        for (e, a), trainer_name in self.map_ea2t.items():
            self.control_agent_count[trainer_name] += 1
            self.control_agents[trainer_name].append((e, a))

        self.active_trainers = []
        self.buffer_pool: Dict[str, SharedReplayBuffer] = {} 
        for trainer_name in self.trainer_pool.keys():
            # set n_rollout_threads as control_agent_count[trainer_name] and num_agents as 1
            if self.control_agent_count[trainer_name] > 0:
                policy_args, obs_space, share_obs_space, act_space = self.policy_config(trainer_name)
                self.buffer_pool[trainer_name] = SharedReplayBuffer(
                    policy_args, 1, obs_space, share_obs_space, act_space, 
                    n_rollout_threads=self.control_agent_count[trainer_name])
                self.trainer_pool[trainer_name].to(self.device)
                self.active_trainers.append(trainer_name)
            else:
                if load_unused_to_cpu:
                    self.trainer_pool[trainer_name].to(torch.device("cpu"))
                else:
                    self.trainer_pool[trainer_name].to(self.device)
                self.buffer_pool[trainer_name] = None
        #print("active trainers:", self.active_trainers)
        self.__initialized = True
    
    def extract_elements(self, trainer_name, x):
        return np.stack([x[e][a] for e, a in self.control_agents[trainer_name]])

    def skip(self, trainer_name):
        # produce actions in parallel envs, skip this trainer 
        return (self.use_policy_in_env and trainer_name not in self.on_training) or (trainer_name.startswith("script:"))

    def init_first_step(self, share_obs:np.ndarray, obs:np.ndarray):
        assert self.__initialized
        for trainer_name in self.active_trainers:
            # extract corresponding (e, a) and add num_agent=1 dimension
            obs_lst = np.expand_dims(self.extract_elements(trainer_name, obs), axis=1)
            share_obs_lst = np.expand_dims(self.extract_elements(trainer_name, share_obs), axis=1)
            self.buffer_pool[trainer_name].share_obs[0] = share_obs_lst.copy()
            self.buffer_pool[trainer_name].obs[0] = obs_lst.copy()
        self._step = 0
    
    def reward_shaping_steps(self):
        """This should differ among algorithms and should be overrided by subclasses.
        """
        reward_shaping_steps = []
        for e in range(self.n_rollout_threads):
            train_tot_num_steps = [self.trainer_total_num_steps[self.map_ea2t[(e, a)]] * int(self.map_ea2t[(e, a)] in self.on_training) for a in range(self.num_agents)]
            reward_shaping_steps.append(max(train_tot_num_steps))
        return reward_shaping_steps

    @torch.no_grad()
    def step(self, step):
        assert self.__initialized
        actions = np.full((self.n_rollout_threads, self.num_agents), fill_value=None).tolist()
        self.step_data = dict()
        for trainer_name in self.active_trainers:
            self.trainer_total_num_steps[trainer_name] += self.control_agent_count[trainer_name]
            self.train_infos[f"{trainer_name}-total_num_steps"] = self.trainer_total_num_steps[trainer_name]

            if self.skip(trainer_name):
                continue

            trainer = self.trainer_pool[trainer_name]
            buffer = self.buffer_pool[trainer_name]

            trainer.prep_rollout()
            
            value, action, action_log_prob, rnn_states, rnn_states_critic \
                = trainer.policy.get_actions(np.concatenate(buffer.share_obs[step]),
                                np.concatenate(buffer.obs[step]),
                                np.concatenate(buffer.rnn_states[step]),
                                np.concatenate(buffer.rnn_states_critic[step]),
                                np.concatenate(buffer.masks[step]))

            value = np.expand_dims(np.array(_t2n(value)), axis=1)
            action = np.expand_dims(np.array(_t2n(action)), axis=1)
            action_log_prob = np.expand_dims(np.array(_t2n(action_log_prob)), axis=1)
            rnn_states = np.expand_dims(np.array(_t2n(rnn_states)), axis=1)
            rnn_states_critic = np.expand_dims(np.array(_t2n(rnn_states_critic)), axis=1)

            self.step_data[trainer_name] = value, action, action_log_prob, rnn_states, rnn_states_critic
            
            for i, (e, a) in enumerate(self.control_agents[trainer_name]):
                actions[e][a] = action[i][0]
        return actions
    
    def insert_data(self, share_obs, obs, rewards, dones, active_masks=None, bad_masks=None, infos=None):
        """
            ndarrays of shape (n_rollout_threads, num_agents, *)
        """
        assert self.__initialized
        self._step += 1
        for trainer_name in self.active_trainers:
            if self.skip(trainer_name):
                continue

            trainer = self.trainer_pool[trainer_name]
            buffer = self.buffer_pool[trainer_name]

            value, action, action_log_prob, rnn_states, rnn_states_critic = self.step_data[trainer_name]

            # (control_agent_count[trainer_name], 1, *)            
            obs_lst = np.expand_dims(self.extract_elements(trainer_name, obs), axis=1)
            share_obs_lst = np.expand_dims(self.extract_elements(trainer_name, share_obs), axis=1)
            rewards_lst = np.expand_dims(self.extract_elements(trainer_name, rewards), axis=1)
            dones_lst = np.expand_dims(self.extract_elements(trainer_name, dones), axis=1)

            rnn_states[dones_lst == True] = np.zeros(((dones_lst == True).sum(), buffer.recurrent_N, buffer.hidden_size), dtype=np.float32)
            rnn_states_critic[dones_lst == True] = np.zeros(((dones_lst == True).sum(), *buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
            masks = np.ones((self.control_agent_count[trainer_name], 1, 1), dtype=np.float32)
            masks[dones_lst == True] = np.zeros(((dones_lst == True).sum(), 1), dtype=np.float32)

            bad_masks_lst = active_masks_lst = None
            if bad_masks is not None:
                bad_masks_lst = np.expand_dims(self.extract_elements(trainer_name, bad_masks), axis=1)
            if active_masks is not None:
                active_masks_lst = np.expand_dims(self.extract_elements(trainer_name, active_masks), axis=1)

            if self.all_args.use_task_v_out:
                value = value[:, :, self.policy_id(trainer_name)][:, :, np.newaxis]

            buffer.insert(share_obs_lst, obs_lst, rnn_states, rnn_states_critic, action, action_log_prob, value, rewards_lst, masks, active_masks=active_masks_lst, bad_masks=bad_masks_lst)

            if infos is not None:
                if self.all_args.env_name == "Overcooked" and self.all_args.predict_other_shaped_info:
                    if not hasattr(buffer, "other_shaped_info"):
                        buffer.other_shaped_info = np.zeros((buffer.episode_length + 1, buffer.n_rollout_threads, 1, 12), dtype=np.int32)
                    for i, (e, a) in enumerate(self.control_agents[trainer_name]):
                        buffer.other_shaped_info[self._step, i, 0] = infos[e]["vec_shaped_info_by_agent"][1-a] # insert other agent's shaped info
            
            # partner policy info
            if self.all_args.env_name == "Overcooked":
                if self.all_args.policy_group_normalization and not hasattr(buffer, "other_policy_type"):
                    buffer.other_policy_type = np.zeros((buffer.episode_length + 1, buffer.n_rollout_threads, 1, 1), dtype=np.int32)
                    for i, (e, a) in enumerate(self.control_agents[trainer_name]):
                        buffer.other_policy_type[:, i, :, :] = self.policy_type(self.map_ea2t[(e, 1-a)])
                
                if not hasattr(buffer, "other_policy_id"):
                    buffer.other_policy_id = np.zeros((buffer.episode_length + 1, buffer.n_rollout_threads, 1, 1), dtype=np.int32)
                    for i, (e, a) in enumerate(self.control_agents[trainer_name]):
                        buffer.other_policy_id[:, i, :, :] = self.policy_id(self.map_ea2t[(e, 1-a)])

        self.step_data = None
    
    def compute_advantages(self):
        all_adv = defaultdict(list)
        for trainer_name in self.active_trainers:
            trainer = self.trainer_pool[trainer_name]
            buffer = self.buffer_pool[trainer_name]

            if trainer_name in self.on_training:
                advantages = trainer.compute_advantages(buffer)
                for i, (e, a) in enumerate(self.control_agents[trainer_name]):
                    all_adv[(self.map_ea2t[(e, 0)], self.map_ea2t[(e, 1)], a)].append(advantages[:, i].mean())
        return all_adv

    def train(self, **kwargs):
        assert self.__initialized
        for trainer_name in self.active_trainers:
            trainer = self.trainer_pool[trainer_name]
            buffer = self.buffer_pool[trainer_name]
            
            if trainer_name in self.on_training:
                trainer.prep_rollout()

                # compute returns
                next_values = trainer.policy.get_values(np.concatenate(buffer.share_obs[-1]),
                                                        np.concatenate(buffer.rnn_states_critic[-1]),
                                                        np.concatenate(buffer.masks[-1]))
                next_values = np.expand_dims(np.array(_t2n(next_values)), axis=1)
                if self.all_args.use_task_v_out:
                    next_values = next_values[:, :, self.policy_id(trainer_name)][:, :, np.newaxis]
                buffer.compute_returns(next_values, trainer.value_normalizer)

                # train
                trainer.prep_training()
                train_info = trainer.train(buffer, turn_on=(self.trainer_total_num_steps[trainer_name] >= self.all_args.critic_warmup_horizon))
                self.train_infos.update({f"{trainer_name}-{k}": v for k, v in train_info.items()})
                self.train_infos.update({f"{trainer_name}-average_episode_rewards": np.mean(buffer.rewards) * buffer.episode_length})

            # place first step observation of next episode 
            buffer.after_update()

        return copy.deepcopy(self.train_infos)
    
    def lr_decay(self, episode, episodes):
        for trainer_name in self.on_training:
            self.trainer_pool[trainer_name].policy.lr_decay(episode, episodes)
        
    def update_best_r(self, d, save_dir=None):
        for trainer_name, r in d.items():
            trainer = self.trainer_pool[trainer_name]
            if r > self.best_r[trainer_name]:
                self.best_r[trainer_name] = r
                if trainer_name in self.on_training and save_dir is not None:
                    if not os.path.exists(str(save_dir) + "/{}".format(trainer_name)):
                        os.makedirs(str(save_dir) + "/{}".format(trainer_name))
                    #print("save", str(save_dir) + "/{}".format(trainer_name), f"best_r")
                    if self.policy_config(trainer_name)[0].use_single_network:
                        policy_model = trainer.policy.model
                        torch.save(policy_model.state_dict(), str(save_dir) + "/{}/model_best_r.pt".format(trainer_name))
                    else:
                        policy_actor = trainer.policy.actor
                        torch.save(policy_actor.state_dict(), str(save_dir) + "/{}/actor_best_r.pt".format(trainer_name))
                        policy_critic = trainer.policy.critic
                        torch.save(policy_critic.state_dict(), str(save_dir) + "/{}/critic_best_r.pt".format(trainer_name))


    def save(self, step, save_dir):
        for trainer_name in self.on_training:
            trainer = self.trainer_pool[trainer_name]
            if not os.path.exists(str(save_dir) + "/{}".format(trainer_name)):
                os.makedirs(str(save_dir) + "/{}".format(trainer_name))
            trainer_step = self.trainer_total_num_steps[trainer_name]
            #print("save", str(save_dir) + "/{}".format(trainer_name), f"periodic_{trainer_step}")
            if self.policy_config(trainer_name)[0].use_single_network:
                policy_model = trainer.policy.model
                torch.save(policy_model.state_dict(), str(save_dir) + "/{}/model_periodic_{}.pt".format(trainer_name, trainer_step))
            else:
                policy_actor = trainer.policy.actor
                torch.save(policy_actor.state_dict(), str(save_dir) + "/{}/actor_periodic_{}.pt".format(trainer_name, trainer_step))
                policy_critic = trainer.policy.critic
                torch.save(policy_critic.state_dict(), str(save_dir) + "/{}/critic_periodic_{}.pt".format(trainer_name, trainer_step))
