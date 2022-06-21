# HSP
This is a repository for Hidden-utility Self-Play.

# Overcooked
Our experiments are conducted in three layouts from [On the Utility of Learning about Humans for Human-AI Coordination](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019), named *Asymmetric Advantages*, *Coordination Ring*, and *Counter Circuit*,  and two designed layouts, named *Distant Tomato* and *Many Orders*.

# Training
Here we take Distant Tomato as an example to show the training procedure.

## Self-Play

To train self-play policies, go to directory `hsp/scripts`, change `layout` to one of "unident_s"(Asymmetric Advantages), "random1"(Coordination Ring), "random3"(Counter Circuit), "distant_tomato"(Distant_Tomato) and "many_orders"(Many Orders) and run `./train_overcooked_sp.sh`.

## MEP
We reimplemented [Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination](https://github.com/ruizhaogit/maximum_entropy_population_based_training) and achieved significant higher episode reward when paired with human proxy models than reported in original paper. 

To train the first stage for MEP, go to directory `hsp/scripts`, change `layout` to `distant_tomato` and run `./train_overcooked_mep_stage_1.sh`.

After the first stage training is done, copy the init/middle/final actor models of each policy to directory `hsp/policy_pool/distant_tomato/mep/s1` and name the models as follows,
```
- distant_tomato
    - mep
        - s1
            - mep1_init_actor.pt
            - mep1_mid_actor.pt
            - mep1_final_actor.pt
            ...
            - mep12_final_actor.pt
```

To perform second stage training, go to directory `hsp/scripts`, change to the correct layout name, and run `./train_overcooked_mep_stage_2.sh`

## HSP
For the first stage, go to directory `hsp/scripts` and run `./hsp/s1/distant_tomato.sh`. After training, put each pair of biased policy and adaptive policy in directory `hsp/policy_pool/distant_tomato/hsp/s1` and set up the correct naming and pathes to these models in `hsp/policy_pool/distant_tomato/hsp/s1/eval.yml`. 

Then we do evaluation for each pair of biased policy and adaptive policy to compute expected event counts. According to event counts, select policies with greedy policy selection scheme described in the paper. 

For the second stage, we first set up the correct yaml file by including all selected biased policies and some policies from MEP policy pool. Run `./hsp/s2/distant_tomato.sh` in directory `hsp/scripts` to do adaptive training.