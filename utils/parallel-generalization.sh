#!/bin/bash
# Run generalization tests on a trained model
# run from project root! (where Readme is)
# Model zip file paths must be defined in models.txt
printf "\n\n =============== TESTING MODEL =================\n\n"
parallel spr-rl :::: utils/networks.txt :::: utils/agent_configs.txt  :::: utils/sim_configs.txt :::: utils/services.txt ::: "10000" ::: "-t" ::: "best" ::: "-m" :::: utils/models.txt ::: "--sim-seed" :::: utils/30seeds.txt
