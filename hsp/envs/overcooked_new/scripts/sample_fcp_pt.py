import argparse
import re
import os

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description = "Collect specific results from target layout runs")
    # parser.add_argument('--layout', type = str, help = "Path to target layout")
    # parser.add_argument('--iter', type = int, nargs='+', default=[10, 100, 199], help = "target id to select, default as [10, 100, 199]")
    
    # args = parser.parse_args()
    
    layout = "/home/yuchao/project/onpolicy/onpolicy/scripts/results/Overcooked/cramped_room/rmappo/fcp-sp-100M/wandb"
    iter = [5, 22, 60]
    names = ["init", "avg", "final"]
    
    # collect run id list
    run_dir_list = os.listdir(layout)
    seed_id = 1
    for run_dir in run_dir_list:
        if os.path.isdir("{}/{}".format(layout, run_dir)) and (run_dir.find("run-") != -1):
            print("process {}".format(run_dir))
            seed_dir = "{}/uniform/seed_{}".format(layout, seed_id)
            
            if not os.path.exists(seed_dir):
                os.mkdir(seed_dir)
            
            for it, na in zip(iter,names):
                os.system("cp {}/{}/files/actor_periodic_{}.pt {}".format(\
                    layout, run_dir, it, seed_dir))
                os.system("mv {}/actor_periodic_{}.pt {}/actor_{}.pt".format(\
                    seed_dir, it, seed_dir, na))
                os.system("cp {}/{}/files/critic_periodic_{}.pt {}".format(\
                    layout, run_dir, it, seed_dir))
                os.system("mv {}/critic_periodic_{}.pt {}/critic_{}.pt".format(\
                    seed_dir, it, seed_dir, na))
            seed_id = seed_id + 1
    
    