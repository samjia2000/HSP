import argparse
import re
import os

if __name__ == '__main__':

    layout = "/home/nfs_data/yuchao/project/onpolicy/onpolicy/scripts/results/Overcooked/random3/rmappo/separated_random/wandb"
    iter = [490]

    # collect run id list
    run_dir_list = os.listdir(layout)
    num = 1
    for run_dir in run_dir_list:
        if os.path.isdir("{}/{}".format(layout, run_dir)) and (run_dir.find("run-") != -1):
            print("process {}".format(run_dir))

            seed_dir = "{}/random".format(layout)
            
            if not os.path.exists(seed_dir):
                os.mkdir(seed_dir)
            
            for it in iter:
                os.system("cp {}/{}/files/actor_agent0_periodic_{}.pt {}".format(\
                    layout, run_dir, it, seed_dir))
                os.system("mv {}/actor_agent0_periodic_{}.pt {}/actor_{}.pt".format(\
                    seed_dir, it, seed_dir, num))
            num += 1
    
    