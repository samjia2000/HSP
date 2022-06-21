import argparse
import re
import os

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description = "Collect specific results from target layout runs")
    # parser.add_argument('--layout', type = str, help = "Path to target layout")
    # parser.add_argument('--iter', type = int, nargs='+', default=[10, 100, 199], help = "target id to select, default as [10, 100, 199]")
    
    # args = parser.parse_args()
    # unident_s = 18 [5, 18, 70]
    # random0 = 36
    # random1 = 30
    # random3 = 45
    # simple = 18
    layout = "/home/nfs_data/yuchao/project/onpolicy/onpolicy/scripts/results/Overcooked/random3/mappo/sp/wandb"
    iter = [5, 50, 499]
    names = ["init", "avg", "final"]
    
    # collect run id list
    run_dir_list = os.listdir(layout)
    seed_id = 1
    for run_dir in run_dir_list:
        if os.path.isdir("{}/{}".format(layout, run_dir)) and (run_dir.find("run-") != -1):
            print("process {}".format(run_dir))

            seed_dir = "{}/fcp-mlp/seed_{}".format(layout, seed_id)
            
            if not os.path.exists(seed_dir):
                os.mkdir(seed_dir)
            
            for it, na in zip(iter,names):
                # if it == 26 and ("1p2wkget" in run_dir or "3kaimwsh" in run_dir \
                # or "33cjco1q" in run_dir or "3kaimwsh" in run_dir \
                # or "1usd8rx7" in run_dir or "31fmvkgg" in run_dir \
                # or "2oz0crxs" in run_dir):
                #     print("change to 16....")
                #     it = 16
                os.system("cp {}/{}/files/actor_periodic_{}.pt {}".format(\
                    layout, run_dir, it, seed_dir))
                os.system("mv {}/actor_periodic_{}.pt {}/actor_{}.pt".format(\
                    seed_dir, it, seed_dir, na))
                # os.system("cp {}/{}/files/critic_periodic_{}.pt {}".format(\
                #     layout, run_dir, it, seed_dir))
                # os.system("mv {}/critic_periodic_{}.pt {}/critic_{}.pt".format(\
                #     seed_dir, it, seed_dir, na))
            seed_id = seed_id + 1
    
    