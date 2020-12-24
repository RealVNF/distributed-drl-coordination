![Python package](https://github.com/RealVNF/distributed-drl-coordination/workflows/Python%20package/badge.svg)
# Distributed Service Scaling, Placement, and Routing Using Deep Reinforcement Learning
Scaling, Placement, and Routing Using Deep Reinforcement Learning. Created as an implementation for my master's thesis.

## Installation 
This package requires stable_baselines to work. Before installing, make sure the following packages are installed on the system.


```bash
# On Ubuntu
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
```


The package can then be installed as follows

```bash
# Create a python 3 virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Update pip
pip install -U pip

# Install the DRL package and its requirements
pip install -r requirements.txt
```

## Usage

The inputs available for the DRL agent are placed in the `inputs` folder. 

The `inputs` folder is structured as follows:
```
.
├── Config:
│   ├── drl: contains the configuration YAML files for the DRL agent
│   └── simulator: contains the simulator configuration files defining the traffic patterns
├── networks: contains the network GraphML files.
├── services: contains the configuration files for the defined services and their components
└── traces: contains a trace to dynamically change the traffic pattern during simulations

```

Based on these inputs and after installation, a correct installation of the DRL agent can be checked as follows:

```
$ spr-rl --help

Usage: spr-rl [OPTIONS] NETWORK AGENT_CONFIG SIMULATOR_CONFIG SERVICES TRAINING_DURATION

  SPR-RL DRL Scaling and Placement main executable

Options:
  -s, --seed INTEGER       Set the agent's seed
  -t, --test TEXT          Path to test timestamp and seed
  -a, --append_test        test after training
  -m, --model_path TEXT    path to a model zip file
  -ss, --sim-seed INTEGER  simulator seed
  -b, --best               Select the best agent
  --help                   Show this message and exit.

```
To train an ACKTR DRL agent for 100,000 steps then test it immediately on the Abilene network, run the following example command from within this directory as follows:

```
$ spr-rl inputs/networks/abilene_1-5in-1eg/abilene-in1-rand-cap0-2.graphml inputs/config/drl/acktr/acktr_default_4-env.yaml inputs/config/simulator/mean-10.yaml inputs/services/abc-start_delay0.yaml 100000 -a
```

The training of the agent can be parallelized via the GNU Parallel tool. A helper scripts is already provided in the `utils` folder. The inputs of the agent must be defined in the corresponding `*.txt` files inside the `utils` folder To run the parallel scripts: From the current directory, run the following command:

```
$ ./utils/parallel.sh
```

