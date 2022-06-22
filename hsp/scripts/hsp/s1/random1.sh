#!/bin/bash
env="Overcooked"

layout="random1"
version="old"

num_agents=2
algo="mappo"
exp="hsp-S1"
seed_max=36

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_overcooked_hsp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
     --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 \
     --ppo_epoch 15 --wandb_name "wandb_name" --user_name "user_name" --reward_shaping_horizon 0 \
     --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --save_interval 25 --log_inerval 10 --use_recurrent_policy\
     --overcooked_version ${version} \
     --use_hsp --w0 "0,0,0,0,r[-10:10:3],0,r[0:10:2],0,0,r[-10:10:3],r[-10:10:3],r[-10:0:2],0,r[0:1:2]" --w1 "0,0,0,0,0,0,0,0,0,0,0,0,0,1" --random_index --share_policy 
done
