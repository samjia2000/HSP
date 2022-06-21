import argparse
import re
import os

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description = "Collect specific results from target layout runs")
    # parser.add_argument('--layout', type = str, help = "Path to target layout")
    # parser.add_argument('--iter', type = int, nargs='+', default=[10, 100, 199], help = "target id to select, default as [10, 100, 199]")
    
    # args = parser.parse_args()

# TODO: read information from wandb-metadata.json

# (alice_iter, pop_iter)
    ITER_DICT = {
        # 'asymmetric_advantages': (
        # 'unident_s': (
        #     [57600, 1209600, 29952000],
        #     [11200, 1086400, 4468800]
        # ),
        # 'coordination_ring': (
        # 'random1': (
        #     [57600, 3513600, 29952000],
        #     [11200, 705600, 4177600]
        # ),
        # 'counter_circuit_o_1order': (
        # 'random3': (
        #     [57600, 5241600, 29952000],
        #     [11200, 593600, 3976000]
        # ),
        # 'cramped_room': (
        'simple': (
            [57600, 1785600, 29952000],
            [11200, 190400, 3976000]
        ),
    }
    names = ["init", "avg", "final"]

    source_dir = '/home/th/workspace/onpolicy/onpolicy/scripts/results/Overcooked'
    target_dir = '/home/th/workspace/onpolicy/onpolicy/envs/overcooked/policy_pool'
    # target_dir = '/home/th/workspace/policy_pool'

    # sample_types = ['uniform', 'prioritized']
    sample_types = ['']


    run_name = 'pop_11-S1'

    seed_max = 1
    population_size = 11
    for sample_type in sample_types:
        for layout_name, iter in ITER_DICT.items():
            alice_iter, pop_iter = iter
            for seed in range(1, seed_max + 1):
                seed_dir = source_dir + '/' + layout_name + '/traj/' + run_name + '/wandb/' + f'latest-run/files/'
                # seed_target_dir = target_dir + '/' + layout_name + '/traj/' + f'seed{seed}/'
                # seed_target_dir = target_dir + '/' + layout_name + f'/traj_{sample_type}/' + f'seed{seed}/'
                seed_target_dir = target_dir + '/' + layout_name + f'/ppo/traj33/'

                alice_dir = seed_dir + 'alice/'
                alice_target_dir = seed_target_dir + 'alice/'
                if not os.path.exists(alice_target_dir):
                    os.makedirs(alice_target_dir)

                # for it, na in zip(alice_iter, names):
                it = alice_iter[-1]
                na = names[-1]
                # alice_actor_path = alice_dir + f'actor_periodic_{it}.pt'
                # alice_actor_target_path = alice_target_dir + f'actor_{na}.pt'
                # os.system(f'cp {alice_actor_path} {alice_actor_target_path}')

                # alice_critic_path = alice_dir + f'critic_periodic_{it}.pt'
                # alice_critic_target_path = alice_target_dir + f'critic_{na}.pt'
                # os.system(f'cp {alice_critic_path} {alice_critic_target_path}')

                for pop in range(1, population_size + 1):
                    pop_dir = seed_dir + f'ppo{pop}/'
                    pop_target_dir = seed_target_dir + f'seed_{pop}/'
                    if not os.path.exists(pop_target_dir):
                        os.makedirs(pop_target_dir)

                    for it, na in zip(pop_iter, names):
                        pop_actor_path = pop_dir + f'actor_periodic_{it}.pt'
                        pop_actor_target_path = pop_target_dir + f'actor_{na}.pt'
                        os.system(f'cp {pop_actor_path} {pop_actor_target_path}')

                    # for it, na in zip(pop_iter, names):
                    #     pop_critic_path = pop_dir + f'critic_periodic_{it}.pt'
                    #     pop_critic_target_path = pop_target_dir + f'critic_{na}.pt'
                    #     os.system(f'cp {pop_critic_path} {pop_critic_target_path}')