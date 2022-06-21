import copy
import logging
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from hsp.algorithms.utils.util import init, check
from hsp.algorithms.population.utils import NullTrainer
from hsp.utils.util import update_linear_schedule
from hsp.envs.overcooked_new.human_aware_rl.human.process_dataframes import get_human_human_trajectories
from hsp.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action

class BehaviorCloningModel(nn.Module):
    def __init__(self, observation_shape, action_shape, mlp_params, **kwargs):
        super().__init__()
        layers = [nn.Linear(np.prod(observation_shape), mlp_params["net_arch"][0]), nn.ReLU()]
        for i in range(1, mlp_params["num_layers"]):
            layers.append(nn.Linear(mlp_params["net_arch"][i-1], mlp_params["net_arch"][i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(mlp_params["net_arch"][-1], action_shape[0]))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.mlp(x)
        return logits

class BehaviorClonePolicy():
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        bc_params = args.bc_params
        self.bc_model = BehaviorCloningModel(**bc_params).to(self.device)
        self.optimizer = torch.optim.Adam(self.bc_model.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    @property
    def model(self):
        return self.bc_model

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)
    
    def to(self, device):
        self.bc_model.to(device)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(self.device).float()
        batch_size = obs.shape[0]

        logits = self.bc_model(obs)
        action_probs = F.softmax(logits, dim=-1)
        if deterministic:
            actions = action_probs.max(dim=-1, keepdim=False).indices
        else:
            sampler = Categorical(action_probs)
            actions = sampler.sample()
        actions = actions.unsqueeze(-1)
        action_log_probs = action_probs.log()
        raise NotImplementedError("BC model get_actions with can_begin_cook and stuck in info not implemented")
        return torch.zeros(batch_size, 1), actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, info=None):
        obs = check(obs).to(self.device).float()
        batch_size = obs.shape[0]

        logits = self.bc_model(obs)
        action_probs = F.softmax(logits, dim=-1)

        '''if stuck_info is not None:
            for i in range(batch_size):
                stuck, history_a = stuck_info[i]
                if stuck:
                    action_probs[i, history_a] = 0
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)'''

        if deterministic:
            actions = action_probs.max(dim=-1, keepdim=False).indices
        else:
            sampler = Categorical(action_probs)
            actions = sampler.sample()
        actions = actions.unsqueeze(-1)
        if info is not None:
            # random when stuck
            for i in range(batch_size):
                a = int(actions[i])
                while info['stuck'][i][0] and a in info['stuck'][i][1]:
                    a = np.random.randint(0, action_probs.shape[-1])
                actions[i] = a
            
            # begin cook when soup is full
            for i in range(batch_size):
                if info['can_begin_cook'][i]:
                    actions[i] = Action.ACTION_TO_INDEX['interact']
        # print("act", actions, action_probs)
        return actions, rnn_states_actor

    def get_values(self, share_obs, rnn_states_critic, masks):
        batch_size = share_obs.shape[0]
        return torch.zeros_like(batch_size, 1)

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        raise NotImplementedError
    
    def load_checkpoint(self, ckpt_path):
        self.bc_model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

    def prep_training(self):
        self.bc_model.train()

    def prep_rollout(self):
        self.bc_model.eval()
    
    def to(self, device):
        self.bc_model.to(device)

def _pad(sequences, maxlen=None, default=0):
    if not maxlen:
        maxlen = max([len(seq) for seq in sequences])
    for seq in sequences:
        pad_len = maxlen - len(seq)
        seq.extend([default]*pad_len)
    return sequences

def load_data(bc_params, verbose=False):
    processed_trajs = get_human_human_trajectories(**bc_params["data_params"], dataset_type=bc_params["bc_type"], silent=not verbose)
    inputs, targets = processed_trajs["ep_states"], processed_trajs["ep_actions"]

    if bc_params['use_lstm']:
        seq_lens = np.array([len(seq) for seq in inputs])
        seq_padded = _pad(inputs, default=np.zeros((len(inputs[0][0],))))
        targets_padded = _pad(targets, default=np.zeros(1))
        seq_t = np.dstack(seq_padded).transpose((2, 0, 1))
        targets_t = np.dstack(targets_padded).transpose((2, 0, 1))
        return seq_t, seq_lens, targets_t
    else:
        return np.vstack(inputs), None, np.vstack(targets)

def _permute(x, p):
    if x is None:
        return None
    y = np.zeros_like(x)
    for idx0, idx1 in enumerate(p):
        y[idx0] = x[idx1]
    return y

class BehaviorCloneTrainer(NullTrainer):
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):
        self.args = args
        self.policy = policy
        self.device = device

        self.bc_params = args.bc_params

        inputs, seq_lens, targets = load_data(self.bc_params, verbose=True)

        self.training_params = self.bc_params["training_params"]
        self.batch_size = self.training_params["batch_size"]

        self.num_states = inputs.shape[0]
        rand = torch.randperm(self.num_states).numpy()

        inputs = _permute(inputs, rand)
        targets = _permute(targets, rand)

        self.num_actions = len(Action.ALL_ACTIONS)

        self.num_valid = int(self.training_params["validation_split"] * self.num_states)
        self.num_train = self.num_states - self.num_valid

        self.validation_data = {
            "inputs": inputs[:self.num_valid],
            "targets": targets[:self.num_valid]
        }
        self.compute_weights(self.validation_data, "val")

        self.training_data = {
            "inputs": inputs[self.num_valid:],
            "targets": targets[self.num_valid:]
        }
        self.compute_weights(self.training_data, "train")

        self.trained_epochs = 0
        self.train_logger = logging.getLogger("BC-Train")
        self.valid_logger = logging.getLogger("BC-Validation")

    def compute_weights(self, data, tag):
        targets = data["targets"]
        weights = np.ones_like(targets, dtype=np.float32)
        act_weight = np.array([1., 1., 1., 1., 1, 1])
        act_weight = act_weight / act_weight.sum(axis=-1)
        for i in range(self.num_actions):
            act_count = (targets == i).astype(np.int32).sum()
            act_ratio = act_count / targets.size
            print(f"[{tag}] Action {Action.INDEX_TO_ACTION[i]}: {act_ratio}")
            weights[targets == i] = act_weight[i] / act_ratio
        data.update(weights=weights)

    def train_one_epoch(self):
        pbar = tqdm(range(int(np.ceil(self.num_train / self.batch_size))), leave=False)
        CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        accuracy = np.zeros((self.num_actions,))
        pred_actions = np.zeros((self.num_train, ))
        sum_loss = 0
        for i in pbar:
            pbar.set_description("")
            l = i * self.batch_size
            r = min((i + 1) * self.batch_size, self.num_train)
            x = check(self.training_data["inputs"][l:r]).to(self.device).float()
            y = check(self.training_data["targets"][l:r]).to(self.device).long().reshape(-1)
            weights = check(self.training_data["weights"][l:r]).to(self.device).float().reshape(-1)
            logits = self.policy.bc_model(x)
            actions = logits.max(dim=-1).indices

            loss = (CrossEntropyLoss(logits, y) * weights).sum() / (r-l)

            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

            sum_loss += loss.item() * (r-l)
            pred_actions[l:r] = actions.cpu().numpy().reshape(-1)
        targets = self.training_data["targets"].reshape(-1)
        for a in range(self.num_actions):
            bingo = ((pred_actions == targets) * (targets == a)).astype(np.int32).sum()
            accuracy[a] = bingo / (targets == a).astype(np.int32).sum()

        '''cook_count = 0
        success_cook_count = 0
        for i in range(self.num_train - 1):
            if self.training_data["inputs"][i+1].max() == 19 and self.training_data["targets"][i].sum() == 5:
                cook_count += 1
                if pred_actions[i] == 5:
                    success_cook_count += 1
        
        print(f"cook: {success_cook_count}/{cook_count}")'''

        self.trained_epochs += 1
        avg_loss = sum_loss / self.num_train
        self.train_logger.info(f"Epoch: {self.trained_epochs}, accuracy: {accuracy}, avg_loss: {avg_loss}")
        train_info = {"accuracy": accuracy, "avg_loss": avg_loss}
        return train_info

    @torch.no_grad()
    def validate(self):
        pbar = tqdm(range(int(np.ceil(self.num_valid / self.batch_size))), leave=False)
        CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        accuracy = np.zeros((self.num_actions,))
        pred_actions = np.zeros((self.num_valid,))
        sum_loss = 0
        for i in pbar:
            l = i * self.batch_size
            r = min((i + 1) * self.batch_size, self.num_valid)

            x = check(self.validation_data["inputs"][l:r]).to(self.device).float()
            y = check(self.validation_data["targets"][l:r]).to(self.device).long().reshape(-1)
            weights = check(self.validation_data["weights"][l:r]).to(self.device).float().reshape(-1)

            logits = self.policy.bc_model(x)
            actions = logits.max(dim=-1).indices

            loss = (CrossEntropyLoss(logits, y) * weights).sum() / (r-l)

            sum_loss += loss.item() * (r-l)
            pred_actions[l:r] = actions.cpu().numpy().reshape(-1)
        targets = self.validation_data["targets"].reshape(-1)
        for a in range(self.num_actions):
            bingo = ((pred_actions == targets) * (targets == a)).astype(np.int32).sum()
            accuracy[a] = bingo / (targets == a).astype(np.int32).sum()
        self.trained_epochs += 1
        avg_loss = sum_loss / self.num_valid
        self.valid_logger.info(f"Epoch: {self.trained_epochs}, accuracy: {accuracy}, avg_loss: {avg_loss}")
        return {"accuracy": accuracy, "avg_loss": avg_loss}

    def prep_training(self):
        self.policy.bc_model.train()

    def prep_rollout(self):
        self.policy.bc_model.eval()
    
    def to(self, device):
        self.policy.to(device)
