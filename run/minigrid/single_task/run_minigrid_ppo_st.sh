#!/bin/bash

cd ../../../

ENV_NAME=$1

# for ENV_NAME in "MiniGrid-DoorKey-6x6-v0" "MiniGrid-DistShift1-v0" "MiniGrid-RedBlueDoors-6x6-v0" "MiniGrid-LavaGapS7-v0" "MiniGrid-MemoryS11-v0" "MiniGrid-SimpleCrossingS9N2-v0" "MiniGrid-MultiRoom-N2-S4-v0"
# do
python run_minigrid_ppo_st.py  --n_exp 30 \
                            --env_name ${ENV_NAME} --exp_name ppo_st_baseline  \
                            --n_epochs 100 --n_steps 2000  --n_episodes_test 16 --train_frequency 2000 --lr_actor 1e-3 --lr_critic 1e-3 \
                            --critic_network MiniGridPPONetwork --critic_n_features 128 \
                            --actor_network MiniGridPPONetwork --actor_n_features 128 \
                            --batch_size 256 --gamma 0.99 --wandb --wandb_entity [WANDB_ENTITY]