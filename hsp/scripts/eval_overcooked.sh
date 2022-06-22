#!/bin/bash
env="Overcooked"

# unident_s, random1, random3, distant_tomato, many_orders
layout="distant_tomato"

version="old"
if [[ "${layout}" == "distant_tomato" || "${layout}" == "many_orders" ]]; then
    version="new"
fi

num_agents=2
algo="population"
agent0_policy_name="agent0_policy_name"
agent1_policy_name="agent1_policy_name"
exp="eval-${agent0_policy_name}-${agent1_policy_name}"

path=../policy_pool

export POLICY_POOL=${path}

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
CUDA_VISIBLE_DEVICES=0 python eval/eval_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} \
--user_name "user_name" --num_agents ${num_agents} --seed 1 --episode_length 400 --n_eval_rollout_threads 1 --eval_episodes 1 --eval_stochastic \
--wandb_name "wandb_name" --use_wandb \
--population_yaml_path ${path}/render_config/${layout}.yml \
--agent0_policy_name ${agent0_policy_name} \
--agent1_policy_name ${agent1_policy_name} --overcooked_version ${version} 