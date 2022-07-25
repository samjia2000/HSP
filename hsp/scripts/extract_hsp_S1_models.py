import glob
import os
import sys
import wandb
import numpy as np

WANDB_NAME="WANDB_NAME"
POLICY_POOL_PATH = "../policy_pool"
RESULT_PATH = "results/Overcooked/{layout}/mappo/hsp-S1/wandb"

def extract_sp_S1_models(layout):
    wandb_dir = RESULT_PATH.format(layout=layout)
    runs = glob.glob(f"{wandb_dir}/run*")
    run_ids = [x.split('-')[-1] for x in runs]
    print(runs)
    print(run_ids)
    api = wandb.Api()
    i = 0
    for run_id in run_ids:
        run = api.run(f"{WANDB_NAME}/Overcooked/{run_id}")
        if run.state == "finished":
            config_str = run.json_config.replace('false', "False").replace("null", "None").replace("true", "True")
            config_str = [x for x in config_str.split(',') if 'seed' in x]
            seed = eval(config_str[0].split(': ')[-1])

            try:
                print("seed", seed)
                w0_actor_pt = run.file("actor_agent0_periodic_249.pt")
                w1_actor_pt = run.file("actor_agent1_periodic_249.pt")
                w0_actor_pt.download("tmp", replace=True)
                w1_actor_pt.download("tmp", replace=True)

                hsp_s1_dir = f"{POLICY_POOL_PATH}/{layout}/hsp/s1"
                os.system(f"mv tmp/actor_agent0_periodic_249.pt {hsp_s1_dir}/hsp{seed}_w0_actor.pt")
                os.system(f"mv tmp/actor_agent1_periodic_249.pt {hsp_s1_dir}/hsp{seed}_w1_actor.pt")
            except:
                pass

if __name__ == "__main__":
    exec('u=1')
    print(u)
    layout = sys.argv[1]
    assert layout in ["random1", "random3", "unident_s", "distant_tomato", "many_orders", "all"]
    if layout == "all":
        for layout in ["random1", "random3", "unident_s", "distant_tomato", "many_orders"]:
            extract_sp_S1_models(layout)
    else:
        extract_sp_S1_models(layout)
