from collections import defaultdict
import numpy as np
import pandas as pd
import sys
import os
import argparse
import glob

def compute_metric(events: dict, event_types: list, layout: str):
    def empty_event_count():
        return {k: 0 for k in event_types}
    ec = defaultdict(empty_event_count)
    for run in events.keys():
        i = eval(run.split('_')[0][3:])
        run_ec = {k[:-3]: v for k, v in events[run].items() if k.endswith('w0')}
        for k in event_types:
            ec[i][k] += run_ec[k]
    runs = sorted(ec.keys())
    print("runs:",runs)
    event_np = np.array([[v for _, v in ec[i].items()] for i in runs])
    df = pd.DataFrame(event_np, index=runs, columns=event_types)

    event_ratio_np = event_np / (event_np.max(axis=0) + 1e-3).reshape(1, -1)

    return runs, event_ratio_np, df

def select_policies(runs, metric_np, K):
    S = []
    n = len(runs)
    S.append(np.random.randint(0, n))
    for iter in range(1, K):
        v = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            if i not in S:
                for j in S:
                    v[i] += abs(metric_np[i] - metric_np[j]).sum()
            else:
                v[i] = -1e9
        x = v.argmax()
        S.append(x)
    S = sorted([runs[i] for i in S])
    return S

def parse_args(args):
    
    parser = argparse.ArgumentParser(
        description='hsp', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--layout", type=str, required=True, help="layout name")
    parser.add_argument("--k", type=int, default=18, help="number of selected policies")
    parser.add_argument("--eval_result_dir", type=str, default="hsp/biased_eval")
    parser.add_argument("--policy_pool_path", type=str, default="../policy_pool")

    args = parser.parse_known_args(args)[0]
    return args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    layout = args.layout
    overcooked_version = "old"
    if layout in ["distant_tomato", "many_orders"]:
        overcooked_version = "new"
    K = args.k

    if overcooked_version == "old":
        event_types = [
                    "put_onion_on_X",
                    # "put_tomato_on_X",
                    "put_dish_on_X",
                    "put_soup_on_X",
                    "pickup_onion_from_X",
                    "pickup_onion_from_O",
                    # "pickup_tomato_from_X",
                    # "pickup_tomato_from_T",
                    "pickup_dish_from_X",
                    "pickup_dish_from_D",
                    "pickup_soup_from_X",
                    "USEFUL_DISH_PICKUP", # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
                    "SOUP_PICKUP", # counted when soup in the pot is picked up (not a soup placed on the table)
                    "PLACEMENT_IN_POT", # counted when some ingredient is put into pot
                    "delivery"
                ]
    else:
        event_types = [
            "put_onion_on_X",
            "put_tomato_on_X",
            "put_dish_on_X",
            "put_soup_on_X",
            "pickup_onion_from_X",
            "pickup_onion_from_O",
            "pickup_tomato_from_X",
            "pickup_tomato_from_T",
            "pickup_dish_from_X",
            "pickup_dish_from_D",
            "pickup_soup_from_X",
            "USEFUL_DISH_PICKUP", # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
            "SOUP_PICKUP", # counted when soup in the pot is picked up (not a soup placed on the table)
            "PLACEMENT_IN_POT", # counted when some ingredient is put into pot
            "viable_placement",
            "optimal_placement",
            "catastrophic_placement",
            "useless_placement",
            "potting_onion",
            "potting_tomato",
            "delivery",
        ]
    events = dict()
    eval_result_dir = os.path.join(args.eval_result_dir, layout)
    logfiles = glob.glob(f"{eval_result_dir}/eval*.txt")
    for logfile in logfiles:
        with open(logfile, 'r') as f:
            lines = [line for line in f.readlines()]
            lines = [line for line in lines if line.startswith("{")]
            for line in lines:
                exec(f"d={line}")
                i = eval(line.split('_')[0][5:])
                exp_name=line.split('-eval')[0][2:]
                d = {k.split(exp_name)[1][1:][8:] : v[0] for k, v in d.items() if k.startswith(exp_name) and any([key in k for key in event_types])}
                if len(d) == 0:
                    continue
                if exp_name == f"hsp{i}_w0-hsp{i}_w1":
                    exp_name=f"hsp{i}_w0-w1"
                    d = {k.replace("by_agent0", "w0").replace("by_agent1", "w1") : v for k, v in d.items()}
                elif exp_name == f"hsp{i}_w1-hsp{i}_w0":
                    exp_name=f"hsp{i}_w1-w0"
                    d = {k.replace("by_agent0", "w1").replace("by_agent1", "w0") : v for k, v in d.items()}
                else:
                    raise RuntimeError(f"Unsupported exp name {exp_name}")
                # print(f"hsp{i}", exp_name, d)

                events[exp_name] = d

    runs, metric_np, df = compute_metric(events, event_types, layout)
    df.to_excel(f"{eval_result_dir}/event_count.xlsx", sheet_name="Sheet1")
    runs = select_policies(runs, metric_np, K)
    print(runs)
    
    # generate HSP training config
    with open(f"{args.policy_pool_path}/{layout}/hsp/s2/train.yml", "w") as f:
        f.write(f"""hsp_adaptive:
    policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
    featurize_type: ppo
    train: True
mep1_1:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep1_init_actor.pt
mep1_2:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep1_mid_actor.pt
mep1_3:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep1_final_actor.pt
mep2_1:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep2_init_actor.pt
mep2_2:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep2_mid_actor.pt
mep2_3:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep2_final_actor.pt
mep3_1:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep3_init_actor.pt
mep3_2:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep3_mid_actor.pt
mep3_3:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep3_final_actor.pt
mep4_1:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep4_init_actor.pt
mep4_2:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep4_mid_actor.pt
mep4_3:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep4_final_actor.pt
mep5_1:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep5_init_actor.pt
mep5_2:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep5_mid_actor.pt
mep5_3:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep5_final_actor.pt
mep6_1:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep6_init_actor.pt
mep6_2:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep6_mid_actor.pt
mep6_3:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/mep/s1/mep6_final_actor.pt\n""")
        for i in runs:
            f.write(f"""hsp{i}:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/hsp/s1/hsp{i}_w0_actor.pt\n""")