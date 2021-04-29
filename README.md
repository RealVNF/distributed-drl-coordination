![Python package](https://github.com/RealVNF/distributed-drl-coordination/workflows/Python%20package/badge.svg)

# Distributed Online Service Coordination Using Deep Reinforcement Learning
Self-learning and self-adaptive service coordination using deep reinforcement learning (DRL). Service coordination includes scaling and placement of chained service components as well as scheduling and routing of flows/requests through the placed instances. We train our proposed DRL approach offline in a centralized fashion and then deploy a distributed DRL agent at each node in the network. This fully distributed DRL approach only requires local observation and control and significantly outperforms existing state-of-the-art solutions.

<p align="center">
  	<img src="docs/logos/realvnf.png" height="100" hspace="50"/>
	<img src="docs/logos/upb.png" height="60" hspace="50"/>
	<img src="docs/logos/huawei.png" height="100" hspace="50"/>
    <img src="docs/logos/swc.png" height="100" hspace="50"/>
</p>

## Citation

If you use this code, please cite our [paper](https://ris.uni-paderborn.de/download/21543/21544/public_author_version.pdf):

```
@inproceedings{schneider2021distributed,
	title={Distributed Online Service Coordination Using Deep Reinforcement Learning},
	author={Schneider, Stefan and Qarawlus, Haydar and Karl, Holger},
	booktitle={IEEE International Conference on Distributed Computing Systems (ICDCS)},
	year={2021},
	organization={IEEE},
	note={to appear}
}
```

## Installation 
This package requires stable_baselines to work. Before installing, make sure the following packages are installed on the system.


```bash
# On Ubuntu
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx libsm6 libxext6
# check your python3 version
python3 --version
```

If your Python version is neither 3.6 nor 3.7 (3.8+ does not support TensorFlow1, which is currently required by [`stable_baselines`](https://github.com/hill-a/stable-baselines)), manually install the correct version as described [here](https://www.techiediaries.com/ubuntu/install-python-3-pip-venv-ubuntu-20-04-19/):

```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7 python3.7-dev
```

The package can then be installed as follows, requiring Python 3.6 or 3.7 :

```bash
# Create a python 3 virtual environment
python3 -m venv venv
# if python3 != 3.6 or 3.7, use the manually installed python3.7 instead (see above)

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

Based on these inputs and after installation, a correct installation of the DRL agent can be checked as follows (ignore `tensorflow` warnings):

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

The results and trained model are saved in the `results` directory, which is created automatically.

### Tensorboard

To visualize training progress, start `tensorboard`:

```
tensorboard --logdir tb
```

Then go to http://localhost:6006

### Parallelization

The training of the agent can be parallelized via the GNU Parallel tool. A helper scripts is already provided in the `utils` folder. The inputs of the agent must be defined in the corresponding `*.txt` files inside the `utils` folder To run the parallel scripts: From the current directory, run the following command:

```
$ ./utils/parallel.sh
```

