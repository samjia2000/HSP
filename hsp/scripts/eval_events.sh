#!/bin/bash
env="Overcooked"

# unident_s, random1, random3, distant_tomato, many_orders
layout=$1

version="old"
if [[ "${layout}" == "distant_tomato" || "${layout}" == "many_orders" ]]; then
    version="new"
fi

num_agents=2
algo="population"

path=../policy_pool

export POLICY_POOL=${path}

echo "env is ${env}, layout is ${layout}, eval"
mkdir -p hsp/biased_eval/${layout}
for i in `seq 36`;
do
    agent0_policy_name="hsp${i}_w0"
    agent1_policy_name="hsp${i}_w1"
    exp="eval-${agent0_policy_name}-${agent1_policy_name}"

    CUDA_VISIBLE_DEVICES=1 python eval/eval_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} \
    --user_name "user_name" --num_agents ${num_agents} --seed 1 --episode_length 400 --n_eval_rollout_threads 100 --eval_episodes 100 --eval_stochastic \
    --wandb_name "wandb_name" --use_wandb \
    --population_yaml_path ${path}/${layout}/hsp/s1/eval.yml \
    --agent0_policy_name ${agent0_policy_name} \
    --agent1_policy_name ${agent1_policy_name} --overcooked_version ${version} >> hsp/biased_eval/${layout}/${exp}.txt

    agent0_policy_name="hsp${i}_w1"
    agent1_policy_name="hsp${i}_w0"
    exp="eval-${agent0_policy_name}-${agent1_policy_name}"

    CUDA_VISIBLE_DEVICES=1 python eval/eval_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} \
    --user_name "user_name" --num_agents ${num_agents} --seed 1 --episode_length 400 --n_eval_rollout_threads 100 --eval_episodes 100 --eval_stochastic \
    --wandb_name "wandb_name" --use_wandb \
    --population_yaml_path ${path}/${layout}/hsp/s1/eval.yml \
    --agent0_policy_name ${agent0_policy_name} \
    --agent1_policy_name ${agent1_policy_name} --overcooked_version ${version} >> hsp/biased_eval/${layout}/${exp}.txt
    
done