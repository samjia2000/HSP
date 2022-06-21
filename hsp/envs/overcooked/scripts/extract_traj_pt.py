import argparse
import re
import os

LAYOUTS = ['unident_s', 'random1', 'random3', 'simple']

# run_name = 'o19-S2-Prioritized-prioritized'
run_name = 'o19-S2-Uniform-uniform'

# sample_type = 'prioritized'
sample_type = 'uniform'

source_dir = '/home/th/workspace/onpolicy/onpolicy/scripts/results/Overcooked/'

periodic = '49996800'

for layout in LAYOUTS:
    target_dir = f'/home/th/workspace/onpolicy/onpolicy/envs/overcooked/policy_pool/{layout}/ppo/alice/traj_{sample_type}/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    alice_path = source_dir + f'{layout}/traj/{run_name}/wandb/latest-run/files/alice/actor_periodic_{periodic}.pt'
    target_path = target_dir + 'actor.pt'
    os.system(f'cp {alice_path} {target_path}')