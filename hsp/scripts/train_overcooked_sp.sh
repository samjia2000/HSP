#!/bin/bash
env="Overcooked"

# unident_s, random1, random3, distant_tomato, many_orders
layout=$1

version="old"
if [[ "${layout}" == "distant_tomato" || "${layout}" == "many_orders" ]]; then
    version="new"
fi

num_agents=2
algo="mappo"
exp="sp"
seed_max=12

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_overcooked_sp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
     --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
     --ppo_epoch 15 \
     --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --save_interval 25 --log_inerval 10 --use_recurrent_policy\
     --overcooked_version ${version} \
     --wandb_name "WANDB_NAME" --user_name "USER_NAME" 
done