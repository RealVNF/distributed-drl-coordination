#!/bin/bash
# Run multiple agents in parallel using GNU Parallel
# run from project root! (where Readme is)
printf "\n\n =============== START TRAINING ================= \n\n"

parallel spr-rl :::: utils/networks.txt :::: utils/agent_configs.txt  :::: utils/sim_configs.txt :::: utils/services.txt ::: "1000000" ::: "-a" ::: "--seed" :::: utils/10seeds.txt

printf "\n\n =============== TRAINING DONE ================= \n\n"
# Select best agent among trained agents
printf "\n\n =============== TESTING BEST AGENT =================\n\n"
parallel spr-rl :::: utils/networks.txt :::: utils/agent_configs.txt  :::: utils/sim_configs.txt :::: utils/services.txt ::: "10000" ::: "--best" ::: "--sim-seed" :::: utils/30seeds.txt
