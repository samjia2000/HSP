# HSP
This is a repository for Hidden-utility Self-Play.

# Installation


```
conda create -n hsp
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
cd hsp
pip install -e . 
pip install wandb icecream setproctitle gym seaborn tensorboardX slackweb psutil slackweb pyastar2d einops
```

We use [wandb](https://wandb.ai) to monitor logs. See the the [official website](https://wandb.ai) and the code for some examples.

# Overcooked
Our experiments are conducted in three layouts from [On the Utility of Learning about Humans for Human-AI Coordination](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019), named *Asymmetric Advantages*, *Coordination Ring*, and *Counter Circuit*,  and two designed layouts, named *Distant Tomato* and *Many Orders*. These layouts are named "unident_s", "random1", "random3", "distant_tomato" and "many_orders" respectively in the code.

# Training

All training scripts are under directory `hsp/scripts`. All methods consist of two stages, in the first of which a pool of policies are trained and in the second of which an adaptive policy is trained against this policy pool. 

## Self-Play

To train self-play policies, change `layout` to one of "unident_s"(Asymmetric Advantages), "random1"(Coordination Ring), "random3"(Counter Circuit), "distant_tomato"(Distant_Tomato) and "many_orders"(Many Orders) and run `./train_overcooked_sp.sh`.

## FCP

In the first stage, run `./train_sp_all_S1.sh` to train 12 polcicies via self-play on each layout. After the first stage training is done, run `python extract_sp_S1_models.py` to extract init, middle and final checkpoints of the self-play policies into the policy pool. At this step, the policy pools of FCP on all layouts should be in the directory `hsp/policy_pool/LAYOUT/fcp/s1`. 

In the second stage, run `./train_fcp_all_S2.sh` to train an adaptive policy against the policy pool for each layout.

## MEP
We reimplemented [Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination](https://github.com/ruizhaogit/maximum_entropy_population_based_training) and achieved significant higher episode reward when paired with human proxy models than reported in original paper. 

For the first stage, run `./train_mep_all_S1.sh`. After training is finished, run `python extract_mep_S1_models.py` to extract checkpoints of the MEP policies into the policy pool. 

For the second stage, run `./train_mep_all_S2.sh`.

## HSP
**Important:** Please make sure you finished the first stage training of MEP before the second stage of HSP.

For the first stage, run `./train_hsp_all_S1.sh`. After training is finished, run `python extract_hsp_S1_models.py` to collect HSP policies into the policy pool. 

Then run `./eval_events_all.sh` to do evaluation to obtain event features for each pair of biased policy and adaptive policy in HSP.  After evaluation is done, for each layout, run `python hsp/greedy_select.py --layout LAYOUT --k 18` to select HSP policies in a greedy manner and generate configuration of policy pool automatically.

For the second stage, run `./train_hsp_all_S2.sh`.

## Evaluation

Run `./eval_overcooked.sh` for evaluation. You can change the layout name, path to YAML file of population configuration and policies to evaluate in `eval_overcooked.sh`. To evaluate with script policies, change policy name to a string with `script:` as prefix, for example, `script:place_onion_and_deliver_soup`. For more script policies, check `script_agent.py` under the overcooked environment directories.

TODO: more detailed evaluation.

# Publication

If you find this repository useful, please [cite our paper](https://openreview.net/forum?id=TrwE8l9aJzs):

```
@inproceedings{
yu2023learning,
title={Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased},
author={Chao Yu and Jiaxuan Gao and Weilin Liu and Botian Xu and Hao Tang and Jiaqi Yang and Yu Wang and Yi Wu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=TrwE8l9aJzs}
}
```
