import argparse
import re
import os

if __name__ == '__main__':

    layout = "/data/home/yuchao/project/onpolicy/onpolicy/scripts/results/Overcooked/coordination_ring/mep/pbt-S1/wandb/run-20220414_135527-nvarvvev/files"
    iter = [400000, 1600000, 10000000]
    names = ["init", "avg", "final"]
    
    # collect run id list
    run_dir_list = os.listdir(layout)

    seed_id = 1
    for run_dir in run_dir_list:
        if os.path.isdir("{}/{}".format(layout, run_dir)) and (run_dir.find("ppo") != -1):
            print("process {}".format(run_dir))
            seed_dir = "{}/mep/seed_{}".format(layout, seed_id)
            
            if not os.path.exists(seed_dir):
                os.mkdir(seed_dir)
            
            for it, na in zip(iter,names):
                if run_dir == "ppo1":
                    it += 40000
                os.system("cp {}/{}/actor_periodic_{}.pt {}".format(\
                    layout, run_dir, it, seed_dir))
                os.system("mv {}/actor_periodic_{}.pt {}/actor_{}.pt".format(\
                    seed_dir, it, seed_dir, na))
                os.system("cp {}/{}/critic_periodic_{}.pt {}".format(\
                    layout, run_dir, it, seed_dir))
                os.system("mv {}/critic_periodic_{}.pt {}/critic_{}.pt".format(\
                    seed_dir, it, seed_dir, na))
            seed_id = seed_id + 1
    
    