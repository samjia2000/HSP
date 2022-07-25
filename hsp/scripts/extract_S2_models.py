import glob
import os
import sys
import wandb
import numpy as np

WANDB_NAME="samji2000"
POLICY_POOL_PATH = "../policy_pool"
RESULT_PATH = "results/Overcooked/{layout}/{ALGO_NAME}/{EXP_NAME}/wandb"

def extract_mep_S1_models(layout, algorithm):
    if algorithm == "hsp":
        ALGO_NAME = "adaptive"
        EXP_NAME = "hsp-S2"
    elif algorithm == "fcp":
        ALGO_NAME = "adaptive"
        EXP_NAME = "fcp-S2"
    elif algorithm == "mep":
        ALGO_NAME = "mep"
        EXP_NAME = "mep-S2"
    else:
        raise KeyError(f"Expected ALGORITHM be one of `hsp`, `mep` and `fcp`. But get {algorithm}.")
    wandb_dir = RESULT_PATH.format(layout=layout, ALGO_NAME=ALGO_NAME, EXP_NAME=EXP_NAME)
    runs = glob.glob(f"{wandb_dir}/run*")
    run_ids = [x.split('-')[-1] for x in runs]
    print(runs)
    print(run_ids)
    api = wandb.Api()
    for i, run_id in enumerate(run_ids):
        run = api.run(f"{WANDB_NAME}/Overcooked/{run_id}")
        if run.state == "finished":
            policy_name = f"{algorithm}_adaptive"
            try:
                ckpt = run.file(f"{policy_name}/actor_periodic_99960000.pt")
                ckpt.download("tmp", replace=True)
                algo_s2_dir = f"{POLICY_POOL_PATH}/{layout}/{algorithm}/s2"
                os.system(f"mv tmp/{policy_name}/actor_periodic_99960000.pt {algo_s2_dir}/{policy_name}.pt")
                print(f"Found adaptive policy for algorithm={algorithm} on layout={layout}")
            except:
                pass

if __name__ == "__main__":
    layout = sys.argv[1]
    assert layout in ["random1", "random3", "unident_s", "distant_tomato", "many_orders", "all"]
    algorithm = sys.argv[2]
    assert algorithm in ["all", "hsp", "mep", "fcp"]
    if layout == "all":
        layout = ["random1", "random3", "unident_s", "distant_tomato", "many_orders"]
    else:
        layout = [layout]
    if algorithm == "all":
        algorithm = ["hsp", "mep", "fcp"]
    else:
        algorithm = [algorithm]

    for l in layout:
        for algo in algorithm:
            extract_mep_S1_models(l, algo)
