#!/bin/bash

cd ../../../

ENV_NAME=$1
N_EXPERTS=$2
LOAD_DIR=$3

python run_minigrid_ppo_tl.py  --n_exp 30 \
                            --env_name ${ENV_NAME} --exp_name ppo_tl_moe_multihead_${N_EXPERTS}e \
                            --n_epochs 100 --n_steps 2000  --n_episodes_test 16 --train_frequency 2000 --lr_actor 1e-3 --lr_critic 1e-3 \
                            --critic_network MiniGridPPOMixtureMHNetwork --critic_n_features 128 --n_experts ${N_EXPERTS} \
                            --actor_network MiniGridPPOMixtureMHNetwork --actor_n_features 128 \
                            --batch_size 256 --gamma 0.99 --wandb --wandb_entity [WANDB_ENTITY] \
                            --load_actor ${LOAD_DIR} \
                            --load_critic ${LOAD_DIR}
