import time
import wandb
import copy
import os
import numpy as np
import itertools
from itertools import chain
import torch
import imageio
import warnings
import functools
from hsp.utils.util import update_linear_schedule
from hsp.runner.shared.base_runner import Runner
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
from collections import defaultdict, deque
from typing import Dict
from icecream import ic
from scipy.stats import rankdata

def _t2n(x):
    return x.detach().cpu().numpy()

class OvercookedRunner(Runner):
    """
    A wrapper to start the RL agent training algorithm.
    """
    def __init__(self, config):
        super(OvercookedRunner, self).__init__(config)
        
        
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes): 
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                obs = np.stack(obs)
                total_num_steps += (self.n_rollout_threads)
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if episode < 50:
                if episode % 1 == 0:
                    self.save(episode)
            elif episode < 100:
                if episode % 2 == 0:
                    self.save(episode)
            else:
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Layout {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.layout_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                env_infos = defaultdict(list)
                if self.env_name == "Overcooked":
                    for info in infos:
                        env_infos['ep_sparse_r_by_agent0'].append(info['episode']['ep_sparse_r_by_agent'][0])
                        env_infos['ep_sparse_r_by_agent1'].append(info['episode']['ep_sparse_r_by_agent'][1])
                        env_infos['ep_shaped_r_by_agent0'].append(info['episode']['ep_shaped_r_by_agent'][0])
                        env_infos['ep_shaped_r_by_agent1'].append(info['episode']['ep_shaped_r_by_agent'][1])
                        env_infos['ep_sparse_r'].append(info['episode']['ep_sparse_r'])
                        env_infos['ep_shaped_r'].append(info['episode']['ep_shaped_r'])

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        obs = np.stack(obs)

        # replay buffer
        if self.use_centralized_V:
            share_obs = share_obs
        else:
            share_obs = obs
        
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
     
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = share_obs
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    def restore(self):
        if self.use_single_network:
            policy_model_state_dict = torch.load(str(self.model_dir) + '/model.pt', map_location=self.device)
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir), map_location=self.device)
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not (self.all_args.use_render or self.all_args.use_eval):
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt', map_location=self.device)
                self.policy.critic.load_state_dict(policy_critic_state_dict)
    
    def save(self, step):
        if self.use_single_network:
            policy_model = self.trainer.policy.model
            torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_periodic_{}.pt".format(step))
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_periodic_{}.pt".format(step))
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_periodic_{}.pt".format(step))
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_env_infos = defaultdict(list)
        eval_average_episode_rewards = []
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_obs = np.stack(eval_obs)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            eval_obs = np.stack(eval_obs)
            eval_average_episode_rewards.append(eval_rewards)
            
            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
        for eval_info in eval_infos:
            eval_env_infos['eval_ep_sparse_r_by_agent0'].append(eval_info['episode']['ep_sparse_r_by_agent'][0])
            eval_env_infos['eval_ep_sparse_r_by_agent1'].append(eval_info['episode']['ep_sparse_r_by_agent'][1])
            eval_env_infos['eval_ep_shaped_r_by_agent0'].append(eval_info['episode']['ep_shaped_r_by_agent'][0])
            eval_env_infos['eval_ep_shaped_r_by_agent1'].append(eval_info['episode']['ep_shaped_r_by_agent'][1])
            eval_env_infos['eval_ep_sparse_r'].append(eval_info['episode']['ep_sparse_r'])
            eval_env_infos['eval_ep_shaped_r'].append(eval_info['episode']['ep_shaped_r'])

        eval_env_infos['eval_average_episode_rewards'] = np.sum(eval_average_episode_rewards, axis=0)
        print("eval average sparse rewards: " + str(np.mean(eval_env_infos['eval_ep_sparse_r'])))
        
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs
        obs, share_obs, available_actions = envs.reset()
        obs = np.stack(obs)

        for episode in range(self.all_args.render_episodes):
            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            trajectory = []
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)
                obs = np.stack(obs)

                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            for info in infos:
                ic(info['episode']['ep_sparse_r_by_agent'][0])
                ic(info['episode']['ep_sparse_r_by_agent'][1])
                ic(info['episode']['ep_shaped_r_by_agent'][0])
                ic(info['episode']['ep_shaped_r_by_agent'][1])
                ic(info['episode']['ep_sparse_r'])
                ic(info['episode']['ep_shaped_r'])

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

    def train_bc_model(self):
        from hsp.algorithms.population.policy_pool import PolicyPool
        # set up policy pool for evaluation
        policy_pool = PolicyPool(*self.policy_config, device=self.device)
        policy_pool.register_policy("default", self.policy, self.policy_config, False)
        policy_pool.trans_to_eval()
        policy_pool.set_map_ea2p({(e, a): "default" for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)})
        self.bc_params = self.all_args.bc_params
        epochs = self.bc_params["training_params"]["epochs"]
        pbar = tqdm(range(1, epochs + 1), leave=True, desc="BC-Train")
        best_acc = 0.
        best_sparse_r = -1e9
        best_sparse_r_agent0 = -1e9
        best_sparse_r_agent1 = -1e9
        sparse_r = 0.
        sparse_r_agent0 = 0.
        sparse_r_agent1 = 0.
        for e in pbar:
            self.trainer.prep_training()
            train_info = self.trainer.train_one_epoch()
            valid_info = self.trainer.validate()

            info = dict([('train_'+k, v) for k, v in train_info.items()] + [('valid_'+k, v) for k, v in valid_info.items()])
            
            # self.log_train(info, total_num_steps=e + 1)

            # best 'interact' accuracy
            if valid_info['accuracy'][-1] > best_acc:
                best_acc = valid_info['accuracy'][-1]
                policy_model = self.trainer.policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_best_val_acc.pt")
            
            # best episode sparse reward
            if e % self.eval_interval == 0:
                eval_results = self.evaluate_with_multi_policy(policy_pool.policy_pool, policy_pool.map_ea2p)
                [sparse_r_agent0] = eval_results["default-default-eval_ep_sparse_r_by_agent0"]
                [sparse_r_agent1] = eval_results["default-default-eval_ep_sparse_r_by_agent1"]
                [sparse_r] = eval_results["default-default-eval_ep_sparse_r"]
                if sparse_r > best_sparse_r:
                    best_sparse_r = sparse_r
                    policy_model = self.trainer.policy.model
                    torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_best_sparse_r.pt")
                if sparse_r_agent0 > best_sparse_r_agent0:
                    best_sparse_r_agent0 = sparse_r_agent0
                    policy_model = self.trainer.policy.model
                    torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_best_sparse_r_agent0.pt")
                if sparse_r_agent1 > best_sparse_r_agent1:
                    best_sparse_r_agent1 = sparse_r_agent1
                    policy_model = self.trainer.policy.model
                    torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_best_sparse_r_agent1.pt")

            if (e % self.save_interval == 0 or e == epochs):
                torch.save(policy_model.state_dict(), str(self.save_dir) + f"/model_epoch{e}.pt")
            
            pbar.set_description("Epoch {}: sparse_r={}, loss={}, accuracy={}, val_loss={}, val_accuracy={}".format(e, sparse_r, train_info['avg_loss'], train_info['accuracy'], valid_info['avg_loss'], valid_info['accuracy']))
        ic(best_acc)
        ic(best_sparse_r)
        ic(best_sparse_r_agent0)
        ic(best_sparse_r_agent1)
        ic(info)

    def evaluate_one_episode_with_multi_policy(self, policy_pool: Dict, map_ea2p: Dict):
        """Evaluate one episode with different policy for each agent.
        Params:
            policy_pool (Dict): a pool of policies. Each policy should support methods 'step' that returns actions given observation while maintaining hidden states on its own, and 'reset' that resets the hidden state.
            map_ea2p (Dict): a mapping from (env_id, agent_id) to policy name
        """
        warnings.warn("Evaluation with multi policy is not compatible with async done.")
        [policy.reset(self.n_eval_rollout_threads, self.num_agents) for policy_name, policy in policy_pool.items()]
        for e in range(self.n_eval_rollout_threads):
            for agent_id in range(self.num_agents):
                if not map_ea2p[(e, agent_id)].startswith("script:"):
                    policy_pool[map_ea2p[(e, agent_id)]].register_control_agent(e, agent_id)

        eval_env_infos = defaultdict(list)
        reset_choose = np.ones(self.n_eval_rollout_threads) == 1
        eval_obs, _, _ = self.eval_envs.reset(reset_choose)

        extract_info_keys = [] # ['stuck', 'can_begin_cook']
        infos = None
        for eval_step in range(self.all_args.episode_length):
            # print("Step", eval_step)
            eval_actions = np.full((self.n_eval_rollout_threads, self.num_agents, 1), fill_value=0).tolist()
            for policy_name, policy in policy_pool.items():
                if len(policy.control_agents) > 0:
                    policy.prep_rollout()
                    policy.to(self.device)
                    obs_lst = [eval_obs[e][a] for (e, a) in policy.control_agents]
                    info_lst = None
                    if infos is not None:
                        info_lst = {k: [infos[e][k][a] for e, a in policy.control_agents] for k in extract_info_keys}
                    agents = policy.control_agents
                    actions = policy.step(np.stack(obs_lst, axis=0), agents, info = info_lst, deterministic = not self.all_args.eval_stochastic)
                    for action, (e, a) in zip(actions, agents):
                        eval_actions[e][a] = action
            # Observe reward and next obs
            eval_actions = np.array(eval_actions)
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)

            infos = eval_infos

        if self.all_args.overcooked_version == "old":
            shaped_info_keys = [
                "put_onion_on_X",
                # "put_tomato_on_X",
                "put_dish_on_X",
                "put_soup_on_X",
                "pickup_onion_from_X",
                "pickup_onion_from_O",
                # "pickup_tomato_from_X",
                # "pickup_tomato_from_T",
                "pickup_dish_from_X",
                "pickup_dish_from_D",
                "pickup_soup_from_X",
                "USEFUL_DISH_PICKUP", # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
                "SOUP_PICKUP", # counted when soup in the pot is picked up (not a soup placed on the table)
                "PLACEMENT_IN_POT", # counted when some ingredient is put into pot
                "delivery"
            ]
        else:
            shaped_info_keys = [
                "put_onion_on_X",
                "put_tomato_on_X",
                "put_dish_on_X",
                "put_soup_on_X",
                "pickup_onion_from_X",
                "pickup_onion_from_O",
                "pickup_tomato_from_X",
                "pickup_tomato_from_T",
                "pickup_dish_from_X",
                "pickup_dish_from_D",
                "pickup_soup_from_X",
                "USEFUL_DISH_PICKUP", # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
                "SOUP_PICKUP", # counted when soup in the pot is picked up (not a soup placed on the table)
                "PLACEMENT_IN_POT", # counted when some ingredient is put into pot
                "viable_placement",
                "optimal_placement",
                "catastrophic_placement",
                "useless_placement",
                "potting_onion",
                "potting_tomato",
                "delivery",
            ]

        for eval_info in eval_infos:
            for a in range(self.num_agents):
                for i, k in enumerate(shaped_info_keys):
                    eval_env_infos[f'eval_ep_{k}_by_agent{a}'].append(eval_info['episode']['ep_category_r_by_agent'][a][i])
                eval_env_infos[f'eval_ep_sparse_r_by_agent{a}'].append(eval_info['episode']['ep_sparse_r_by_agent'][a])
                eval_env_infos[f'eval_ep_shaped_r_by_agent{a}'].append(eval_info['episode']['ep_shaped_r_by_agent'][a])
            eval_env_infos['eval_ep_sparse_r'].append(eval_info['episode']['ep_sparse_r'])
            eval_env_infos['eval_ep_shaped_r'].append(eval_info['episode']['ep_shaped_r'])
        
        # print(eval_env_infos)

        return eval_env_infos
    
    def evaluate_with_multi_policy(self, policy_pool = None, map_ea2p = None, num_eval_episodes = None):
        """Evaluate with different policy for each agent.
        """
        policy_pool = policy_pool or self.policy.policy_pool
        map_ea2p = map_ea2p or self.policy.map_ea2p
        num_eval_episodes = num_eval_episodes or self.all_args.eval_episodes
        eval_infos = defaultdict(list)
        for episode in range(num_eval_episodes // self.n_eval_rollout_threads):
            eval_env_info = self.evaluate_one_episode_with_multi_policy(policy_pool, map_ea2p)
            for k, v in eval_env_info.items():
                for e in range(self.n_eval_rollout_threads):
                    agent0, agent1 = map_ea2p[(e, 0)], map_ea2p[(e, 1)]
                    for log_name in [f"{agent0}-{agent1}-{k}", f"agent0-{agent0}-{k}", f"agent1-{agent1}-{k}", f"either-{agent0}-{k}", f"either-{agent1}-{k}"]:
                        eval_infos[log_name].append(v[e])
        eval_infos = {k: [np.mean(v),] for k, v in eval_infos.items()}

        print(eval_infos)
        print({k: v for k, v in eval_infos.items() if 'ep_sparse_r' in k})

        return eval_infos
        
    def naive_train_with_multi_policy(self, reset_map_ea2t_fn = None, reset_map_ea2p_fn = None):
        """This is a naive training loop using TrainerPool and PolicyPool. 

        To use PolicyPool and TrainerPool, you should first initialize population in policy_pool, with either:
        >>> self.policy.load_population(population_yaml_path)
        >>> self.trainer.init_population()
        or:
        >>> # mannually register policies
        >>> self.policy.register_policy(policy_name="ppo1", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.policy.register_policy(policy_name="ppo2", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.trainer.init_population()

        To bind (env_id, agent_id) to different trainers and policies:
        >>> map_ea2t = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_rollout_threads) for a in range(self.num_agents)}
        >>> map_ea2p = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)}
        >>> self.trainer.set_map_ea2t(map_ea2t)
        >>> self.policy.set_map_ea2p(map_ea2p)

        Note that map_ea2t is for training while map_ea2p is for policy evaluations

        WARNING: Currently do not support changing map_ea2t and map_ea2p when training. To implement this, we should take the first obs of next episode in the previous buffers and feed into the next buffers.
        """
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0
        env_infos = defaultdict(list)
        self.eval_info = dict()
        self.env_info = dict()

        for episode in range(0, episodes): 
            self.total_num_steps = total_num_steps
            if self.use_linear_lr_decay:
                self.trainer.lr_decay(episode, episodes)
            
            # reset env agents
            if reset_map_ea2t_fn is not None:
                map_ea2t = reset_map_ea2t_fn(episode)
                self.trainer.reset(map_ea2t, self.n_rollout_threads, self.num_agents, n_repeats=None, load_unused_to_cpu=True)
                if self.all_args.use_policy_in_env:
                    load_policy_cfg = np.full((self.n_rollout_threads, self.num_agents), fill_value=None).tolist()
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            trainer_name = map_ea2t[(e, a)]
                            if trainer_name not in self.trainer.on_training:
                                load_policy_cfg[e][a] = self.trainer.policy_pool.policy_info[trainer_name]
                    self.envs.load_policy(load_policy_cfg)

            # init env
            obs, share_obs, available_actions = self.envs.reset()

            # replay buffer
            if self.use_centralized_V:
                share_obs = share_obs
            else:
                share_obs = obs

            self.trainer.init_first_step(share_obs, obs)

            for step in range(self.episode_length):
                # Sample actions
                actions = self.trainer.step(step)
                    
                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor(self.trainer.reward_shaping_steps())

                self.trainer.insert_data(share_obs, obs, rewards, dones, infos=infos)

            # update env infos
            episode_env_infos = defaultdict(list)
            if self.env_name == "Overcooked":
                for e, info in enumerate(infos):
                    agent0_trainer = self.trainer.map_ea2t[(e, 0)]
                    agent1_trainer = self.trainer.map_ea2t[(e, 1)]
                    for log_name in [f"{agent0_trainer}-{agent1_trainer}", f"agent0-{agent0_trainer}", f"agent1-{agent1_trainer}", f"either-{agent0_trainer}", f"either-{agent1_trainer}"]:
                        episode_env_infos[f'{log_name}-ep_sparse_r_by_agent0'].append(info['episode']['ep_sparse_r_by_agent'][0])
                        episode_env_infos[f'{log_name}-ep_sparse_r_by_agent1'].append(info['episode']['ep_sparse_r_by_agent'][1])
                        episode_env_infos[f'{log_name}-ep_shaped_r_by_agent0'].append(info['episode']['ep_shaped_r_by_agent'][0])
                        episode_env_infos[f'{log_name}-ep_shaped_r_by_agent1'].append(info['episode']['ep_shaped_r_by_agent'][1])
                        episode_env_infos[f'{log_name}-ep_sparse_r'].append(info['episode']['ep_sparse_r'])
                        episode_env_infos[f'{log_name}-ep_shaped_r'].append(info['episode']['ep_shaped_r'])
                env_infos.update(episode_env_infos)
            self.env_info.update(env_infos)
            
            # compute return and update network
            train_infos = self.trainer.train(sp_size=getattr(self, "n_repeats", 0)*self.num_agents)
            
            # update advantage moving average
            if not hasattr(self, "avg_adv"):
                self.avg_adv = defaultdict(float)
            adv = self.trainer.compute_advantages()
            for (agent0, agent1, a), vs in adv.items():
                agent_pair = (agent0, agent1)
                for v in vs:
                    if agent_pair not in self.avg_adv.keys():
                        self.avg_adv[agent_pair] = v
                    else:
                        self.avg_adv[agent_pair] = self.avg_adv[agent_pair] * 0.99 + v * 0.01
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if episode < 50:
                if episode % 1 == 0:
                    self.trainer.save(episode, save_dir=self.save_dir)
            elif episode < 500:
                if episode % 2 == 0:
                    self.trainer.save(episode, save_dir=self.save_dir)
            else:
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    self.trainer.save(episode, save_dir=self.save_dir)

            self.trainer.update_best_r({
                trainer_name: np.mean(self.env_info.get(f'either-{trainer_name}-ep_sparse_r', -1e9))
                for trainer_name in self.trainer.active_trainers
            }, save_dir=self.save_dir)

            # log information
            end = time.time()
            print("\n Layout {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.all_args.layout_name,
                            self.algorithm_name,
                            self.experiment_name,
                            episode,
                            episodes,
                            total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start))))
            print("average episode rewards is {}".format({k.split('-')[0]: train_infos[k] 
                for k in train_infos.keys() if "average_episode_rewards" in k}))
            if self.all_args.algorithm_name == 'traj':
                if self.all_args.traj_stage == 1:
                    print(f'jsd is {train_infos["average_jsd"]}')
                    print(f'jsd loss is {train_infos["average_jsd_loss"]}')

            if episode % self.log_interval == 0:
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                if reset_map_ea2p_fn is not None:
                    map_ea2p = reset_map_ea2p_fn(episode)
                    self.policy.set_map_ea2p(map_ea2p, load_unused_to_cpu=True)
                eval_info = self.evaluate_with_multi_policy()
                self.log_env(eval_info, total_num_steps)
                self.eval_info.update(eval_info)
            
            import sys
            sys.stdout.flush()
                 
    def train_fcp(self):
        raise NotImplementedError
    
    def train_mep(self):
        
        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(self.trainer.population.keys()) # Note index and trainer name would not match when there are >= 10 agents

        print(f"population_size: {self.all_args.population_size}, {len(self.trainer.population)}")

        if self.all_args.stage == 1:
            # Stage 1: train a maximum entropy population
            if self.use_eval:
                assert self.n_eval_rollout_threads % self.population_size == 0
                self.all_args.eval_episodes *= self.population_size
                map_ea2p = {(e, a): self.population[e % self.population_size] for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)}
                self.policy.set_map_ea2p(map_ea2p)

            def pbt_reset_map_ea2t_fn(episode):
                # Round robin trainer
                trainer_name = self.population[episode % self.population_size]
                map_ea2t = {(e, a) : trainer_name for e in range(self.n_rollout_threads) for a in range(self.num_agents)}
                return map_ea2t
            
            self.num_env_steps *= self.population_size
            self.save_interval *= self.population_size
            self.log_interval *= self.population_size
            self.eval_interval *= self.population_size

            self.naive_train_with_multi_policy(reset_map_ea2t_fn=pbt_reset_map_ea2t_fn)

            self.all_args.eval_episodes /= self.population_size
            self.num_env_steps /= self.population_size
            self.save_interval /= self.population_size
            self.log_interval /= self.population_size
            self.eval_interval /= self.population_size
        else:
            # Stage 2: train an agent against population with prioritized sampling
            agent_name = self.trainer.agent_name
            assert (self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0 and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0)
            assert self.n_rollout_threads % self.all_args.train_env_batch == 0
            self.all_args.eval_episodes = self.all_args.eval_episodes * self.n_eval_rollout_threads // self.all_args.eval_env_batch
            self.eval_idx = 0
            all_agent_pairs = list(itertools.product(self.population, [agent_name])) + list(itertools.product([agent_name], self.population))
            print(f"all agent pairs: {all_agent_pairs}")

            running_avg_r = -np.ones((self.population_size * 2,), dtype=np.float32) * 1e9

            def mep_reset_map_ea2t_fn(episode):
                # Randomly select agents from population to be trained
                # 1) consistent with MEP to train against one agent each episode 2) sample different agents to train against
                sampling_prob_np = np.ones((self.population_size * 2, )) / self.population_size / 2
                if self.all_args.use_advantage_prioritized_sampling:
                    print("use advantage prioritized sampling")
                    if episode > 0:
                        metric_np = np.array([self.avg_adv[agent_pair] for agent_pair in all_agent_pairs])
                        sampling_rank_np = rankdata(metric_np, method= 'dense')
                        sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                        sampling_prob_np = sampling_prob_np
                        sampling_prob_np /= sampling_prob_np.sum()
                        maxv = 1. / (self.population_size * 2) * 10
                        while sampling_prob_np.max() > maxv + 1e-6:
                            sampling_prob_np = sampling_prob_np.clip(max = maxv)
                            sampling_prob_np /= sampling_prob_np.sum()
                    # print("sampling prob", sampling_prob_np)
                elif self.all_args.mep_use_prioritized_sampling:
                    metric_np = np.zeros((self.population_size * 2,))
                    for i, agent_pair in enumerate(all_agent_pairs):
                        train_r = np.mean(self.env_info.get(f'{agent_pair[0]}-{agent_pair[1]}-ep_sparse_r', -1e9))
                        eval_r = np.mean(self.eval_info.get(f'{agent_pair[0]}-{agent_pair[1]}-eval_ep_sparse_r', -1e9))

                        avg_r = 0.
                        cnt_r = 0
                        if train_r > -1e9:
                            avg_r += train_r * (self.n_rollout_threads // self.all_args.train_env_batch)
                            cnt_r += (self.n_rollout_threads // self.all_args.train_env_batch)
                        if eval_r > -1e9:
                            avg_r += eval_r * (self.all_args.eval_episodes // (self.n_eval_rollout_threads // self.all_args.eval_env_batch))
                            cnt_r += (self.all_args.eval_episodes // (self.n_eval_rollout_threads // self.all_args.eval_env_batch))
                        if cnt_r > 0:
                            avg_r /= cnt_r
                        else:
                            avg_r = -1e9
                        if running_avg_r[i] == -1e9:
                            running_avg_r[i] = avg_r
                        else:
                            # running average
                            running_avg_r[i] = running_avg_r[i] * 0.95 + avg_r * 0.05
                        metric_np[i] = running_avg_r[i]
                    if (metric_np > -1e9).astype(np.int32).sum() > 0:
                        avg_metric = metric_np[metric_np > -1e9].mean()
                    else:
                        # uniform
                        avg_metric = 1.
                    metric_np[metric_np == -1e9] = avg_metric
                    if self.all_args.uniform_preference:
                        pref_count = 0
                        pref_sampling_prob_np = np.zeros_like(metric_np)
                        for i, agent_pair in enumerate(all_agent_pairs):
                            if agent_pair[0].startswith("policy") or agent_pair[1].startswith("policy"):
                                # preference policy
                                metric_np[i] = 0
                                pref_sampling_prob_np[i] = 1.
                                pref_count += 1
                        sampling_rank_np = rankdata(1. / (metric_np + 1e-6), method = 'dense')
                        sampling_rank_np[pref_sampling_prob_np == 1] = 0 # filter out preference policy
                        # print(f'metric_np {metric_np} sampling_rank_np {sampling_rank_np}')
                        mep_sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                        mep_sampling_prob_np = mep_sampling_prob_np ** self.all_args.mep_prioritized_alpha
                        mep_sampling_prob_np /= mep_sampling_prob_np.sum()
                        mep_sampling_prob_np = mep_sampling_prob_np * 0.9 + (mep_sampling_prob_np > 0).astype(np.float32) / (mep_sampling_prob_np > 0).astype(np.int32).sum() * 0.1
                        mep_sampling_prob_np /= mep_sampling_prob_np.sum()
                        pref_sampling_prob_np /= pref_sampling_prob_np.sum()
                        sampling_prob_np = pref_sampling_prob_np * (pref_count / (self.population_size * 2)) + mep_sampling_prob_np* (1. - pref_count / (self.population_size * 2))
                    else:
                        sampling_rank_np = rankdata(1. / (metric_np + 1e-6), method = 'dense')
                        # print(f'metric_np {metric_np} sampling_rank_np {sampling_rank_np}')
                        sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                        sampling_prob_np = sampling_prob_np ** self.all_args.mep_prioritized_alpha
                        sampling_prob_np /= sampling_prob_np.sum()
                assert abs(sampling_prob_np.sum() - 1) < 1e-3

                # log sampling prob
                sampling_prob_dict = {}
                for i, agent_pair in enumerate(all_agent_pairs):
                    sampling_prob_dict[f"sampling_prob/{agent_pair[0]}-{agent_pair[1]}"] = sampling_prob_np[i]
                wandb.log(sampling_prob_dict, step=self.total_num_steps)

                n_selected = self.n_rollout_threads // self.all_args.train_env_batch
                pair_idx = np.random.choice(2 * self.population_size, size = (n_selected,), p = sampling_prob_np)
                if self.all_args.uniform_sampling_repeat > 0:
                    assert n_selected >= 2 * self.population_size * self.all_args.uniform_sampling_repeat
                    i = 0
                    for r in range(self.all_args.uniform_sampling_repeat):
                        for x in range(2 * self.population_size):
                            pair_idx[i] = x
                            i += 1
                map_ea2t = {(e, a): all_agent_pairs[pair_idx[e % n_selected]][a] for e, a in itertools.product(range(self.n_rollout_threads), range(self.num_agents))}
                
                return map_ea2t
            
            def mep_reset_map_ea2p_fn(episode):
                if self.all_args.eval_policy != "":
                    map_ea2p = {(e, a): [self.all_args.eval_policy, agent_name][(e + a) % 2] for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))}
                else:
                    map_ea2p = {(e, a): all_agent_pairs[(self.eval_idx + e // self.all_args.eval_env_batch) % (self.population_size * 2)][a] for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))}
                    self.eval_idx += self.n_eval_rollout_threads // self.all_args.eval_env_batch
                    self.eval_idx %= self.population_size * 2
                featurize_type = [[self.policy.featurize_type[map_ea2p[(e, a)]] for a in range(self.num_agents)]for e in range(self.n_eval_rollout_threads)]
                self.eval_envs.reset_featurize_type(featurize_type)
                return map_ea2p

            self.naive_train_with_multi_policy(reset_map_ea2t_fn=mep_reset_map_ea2t_fn, reset_map_ea2p_fn=mep_reset_map_ea2p_fn)

    def train_traj(self):
        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(self.trainer.population.keys()) # Note index and trainer name would not match when there are >= 10 agents

        print(self.population)

        if self.all_args.traj_stage == 1:

            agent_name = self.all_args.traj_agent_name
            assert (self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0 and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0)
            assert self.n_rollout_threads % self.all_args.train_env_batch == 0
            self.all_args.eval_episodes = self.all_args.eval_episodes * self.n_eval_rollout_threads // self.all_args.eval_env_batch
            self.eval_idx = 0

            self.n_repeat = int(self.n_rollout_threads // (1 + 3 * self.population_size))

            self.xp_pairs = list(itertools.product([agent_name], self.population)) + list(itertools.product(self.population, [agent_name]))
            self.xp_pairs *= self.n_repeat
            self.sp_pairs = [(p, p) for p in self.population] + [(agent_name, agent_name)]
            self.sp_pairs *= self.n_repeat
            self.all_pairs = self.sp_pairs + self.xp_pairs

            # ! suppose every pair rollouts and have same number of corresponding thread
            # assert self.n_rollout_threads == len(self.all_pairs)

            def traj_reset_map_ea2t_fn(episode):
                map_ea2t = {
                    (e, a): self.all_pairs[e][a] for e in range(self.n_rollout_threads) for a in range(self.num_agents)
                }
                return map_ea2t

            def traj_reset_map_ea2p_fn(episode):
                # map_ea2p = {
                #     (e, a): self.all_pairs[e][a] for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)
                # }
                map_ea2p = {
                    (e, a): agent_name for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)
                }
                return map_ea2p

            self.naive_train_with_multi_policy(reset_map_ea2t_fn=traj_reset_map_ea2t_fn, reset_map_ea2p_fn=traj_reset_map_ea2p_fn)
        else:
            agent_name = self.all_args.traj_agent_name
            assert (self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0 and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0)
            assert self.n_rollout_threads % self.all_args.train_env_batch == 0
            self.all_args.eval_episodes = self.all_args.eval_episodes * self.n_eval_rollout_threads // self.all_args.eval_env_batch
            self.eval_idx = 0
            all_agent_pairs = list(itertools.product(self.population, [agent_name])) + list(itertools.product([agent_name], self.population))
            # print(f"all agent pairs: {all_agent_pairs}")
            running_avg_r = -np.ones((self.population_size * 2,), dtype=np.float32) * 1e9

            def traj_reset_map_ea2t_fn(episode):
                # Randomly select agents from population to be trained
                # 1) consistent with Traj to train against one agent each episode 2) sample different agents to train against
                sampling_prob_np = np.ones((self.population_size * 2, )) / self.population_size / 2
                if self.all_args.traj_use_prioritized_sampling:
                    metric_np = np.zeros((self.population_size * 2,))
                    for i, agent_pair in enumerate(all_agent_pairs):
                        train_r = np.mean(self.env_info.get(f'{agent_pair[0]}-{agent_pair[1]}-ep_sparse_r', -1e9))
                        eval_r = np.mean(self.eval_info.get(f'{agent_pair[0]}-{agent_pair[1]}-eval_ep_sparse_r', -1e9))

                        avg_r = 0.
                        cnt_r = 0
                        if train_r > -1e9:
                            avg_r += train_r * (self.n_rollout_threads // self.all_args.train_env_batch)
                            cnt_r += (self.n_rollout_threads // self.all_args.train_env_batch)
                        if eval_r > -1e9:
                            avg_r += eval_r * (self.all_args.eval_episodes // (self.n_eval_rollout_threads // self.all_args.eval_env_batch))
                            cnt_r += (self.all_args.eval_episodes // (self.n_eval_rollout_threads // self.all_args.eval_env_batch))
                        if cnt_r > 0:
                            avg_r /= cnt_r
                        else:
                            avg_r = -1e9
                        if running_avg_r[i] == -1e9:
                            running_avg_r[i] = avg_r
                        else:
                            # running average
                            running_avg_r[i] = running_avg_r[i] * 0.95 + avg_r * 0.05
                        metric_np[i] = running_avg_r[i]
                    if (metric_np > -1e9).astype(np.int32).sum() > 0:
                        avg_metric = metric_np[metric_np > -1e9].mean()
                    else:
                        # uniform
                        avg_metric = 1.
                    metric_np[metric_np == -1e9] = avg_metric
                    sampling_rank_np = rankdata(1. / (metric_np + 1e-6), method = 'dense')
                    # print(f'metric_np {metric_np} sampling_rank_np {sampling_rank_np}')
                    sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                    sampling_prob_np = sampling_prob_np ** self.all_args.traj_prioritized_alpha
                    sampling_prob_np /= sampling_prob_np.sum()
                    sampling_prob_np = 0.1 * np.ones((self.population_size * 2, )) / self.population_size / 2 + 0.9 * sampling_prob_np
                # print(f'sampling_prob_np {sampling_prob_np}')

                n_selected = self.n_rollout_threads // self.all_args.train_env_batch
                pair_idx = np.random.choice(2 * self.population_size, size = (n_selected,), p = sampling_prob_np)
                map_ea2t = {(e, a): all_agent_pairs[pair_idx[e % n_selected]][a] for e, a in itertools.product(range(self.n_rollout_threads), range(self.num_agents))}
                
                return map_ea2t
            
            def traj_reset_map_ea2p_fn(episode):
                map_ea2p = {(e, a): all_agent_pairs[(self.eval_idx + e // self.all_args.eval_env_batch) % (self.population_size * 2)][a] for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))}
                self.eval_idx += self.n_eval_rollout_threads // self.all_args.eval_env_batch
                self.eval_idx %= self.population_size * 2
                return map_ea2p

            self.naive_train_with_multi_policy(reset_map_ea2t_fn=traj_reset_map_ea2t_fn, reset_map_ea2p_fn=traj_reset_map_ea2p_fn)
