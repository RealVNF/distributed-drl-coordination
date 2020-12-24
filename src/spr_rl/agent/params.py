import os
from shutil import copy2
from datetime import datetime
from coordsim.reader.reader import get_config, read_network, network_diameter
from networkx import DiGraph
import pandas as pd


class Params:
    def __init__(
        self,
        seed,
        agent_config,
        sim_config,
        network,
        services,
        training_duration=10000,
        test_mode=None,
        sim_seed=None,
        best=False
    ):
        self.best = best
        # Set the seed of the agent
        self.seed = seed
        self.sim_seed = sim_seed
        # Check to enable test mode
        self.test_mode = test_mode

        # Store paths of config files
        self.agent_config_path = agent_config
        self.sim_config_path = sim_config
        self.services_path = services
        self.network_path = network

        # Get the file stems for result path setup
        self.agent_config_name = os.path.splitext(os.path.basename(self.agent_config_path))[0]
        self.network_name = os.path.splitext(os.path.basename(self.network_path))[0]
        self.services_name = os.path.splitext(os.path.basename(self.services_path))[0]
        self.sim_config_name = os.path.splitext(os.path.basename(self.sim_config_path))[0]

        # Set training and testing durations
        self.training_duration = training_duration

        # Get and load agent configuration file
        self.agent_config = get_config(self.agent_config_path)
        self.testing_duration = self.agent_config.get('testing_duration', 100000)

        # Setup items from agent config file: Episode len, reward_metrics_history
        self.episode_length = self.agent_config['episode_length']  # 1000 arrivals per episode
        self.reward_metrics_history = self.agent_config['reward_history_length']

        # Read the network file, store ingress and egress nodes
        net, self.ing_nodes, self.eg_nodes = read_network(self.network_path)
        self.network: DiGraph = net

        # Get current timestamps - for storing and identifying results
        datetime_obj = datetime.now()
        self.timestamp = datetime_obj.strftime('%Y-%m-%d_%H-%M-%S')
        self.training_id = f"{self.timestamp}_seed{self.seed}"

        # Create results structures
        self.create_result_dir()
        copy2(self.agent_config_path, self.result_dir)
        copy2(self.sim_config_path, self.result_dir)
        copy2(self.network_path, self.result_dir)
        copy2(self.services_path, self.result_dir)

        # ## ACTION AND OBSERVATION SPACE CALCULATIONS ## #

        # Get degree and diameter of network
        self.net_degree = self.get_max_degree()

        # Get network diameter in terms of e2e delay
        self.net_diameter = network_diameter(self.network)

        # Get max link and node cap for all nodes in the network
        self.max_link_caps, self.max_node_cap = self.get_net_max_cap()

        # Observation shape

        # Size of processing element: 1
        self.processing_size = 1
        # Size of distance to egress: 1
        self.dist_to_egress = 1
        # Size of ttl
        self.ttl_size = 1
        # Size of dr observation
        self.dr_size = 1
        # Node resource usage size = this node + max num of neighbor nodes
        self.node_resources_size = 1 + self.net_degree
        # Link resource usage size = max num of neighbor nodes
        self.link_resources_size = self.net_degree
        # Distance of neighbors to egress
        self.neighbor_dist_to_eg = self.net_degree
        # Component availability status = this node + max num of neighbor nodes
        self.vnf_status = 1 + self.net_degree

        # Observation shape = Above elements combined
        self.observation_shape = (
            self.processing_size +
            # self.dist_to_egress +
            self.ttl_size +
            # self.dr_size +
            self.vnf_status +
            self.node_resources_size +
            self.link_resources_size +
            self.neighbor_dist_to_eg,
        )

        # Action space limit (no shape in discrete actions):
        # The possible destinations for the flow = This node + max num of neighbor nodes
        self.action_limit = 1 + self.net_degree

    def get_max_degree(self):
        """ Get the max degree of the network """
        # Init degree to zero
        max_degree = 0
        # Iterate over all nodes in the network
        degrees = []
        for node in self.network.nodes:
            # Get degree of node, compare with current max
            degree = self.network.degree(node)
            degrees.append(degree)
            if degree > max_degree:
                max_degree = degree
        return max_degree

    def get_net_max_cap(self):
        max_link_caps = {}
        max_node_cap = 0.0
        # get neighbor nodes
        for node in self.network.nodes:
            # Check max node caps
            node_cap = self.network.nodes[node]['cap']
            if node_cap > max_node_cap:
                max_node_cap = node_cap
            # Get max link caps
            max_node_link_cap = 0.0
            neighbor_node_ids = list(self.network[node].keys())
            for neighbor_node_id in neighbor_node_ids:
                link_cap = self.network[node][neighbor_node_id]['cap']
                if link_cap > max_node_link_cap:
                    max_node_link_cap = link_cap

            max_link_caps[node] = max_node_link_cap

        return max_link_caps, max_node_cap

    def create_result_dir(self):
        # Set model path
        self.model_path = os.path.join(os.getcwd(), "results", self.agent_config_name,
                                       self.network_name, self.services_name, self.sim_config_name,
                                       self.training_id, "model.zip")
        # Create a result directory structure
        if self.test_mode is not None:
            if self.test_mode is True:
                # We are in append-test
                self.result_dir = os.path.join(os.getcwd(), "results", self.agent_config_name,
                                               self.network_name, self.services_name, self.sim_config_name,
                                               self.training_id, f"t_{self.training_id}")
                self.tb_log_name = os.path.join(self.agent_config_name,
                                                self.network_name, self.services_name, self.sim_config_name,
                                                self.training_id, f"t_{self.training_id}")
            else:
                if self.best:
                    # Must add best to path
                    self.test_mode = self.select_best_agent()
                    self.result_dir = os.path.join(os.getcwd(), "results", self.agent_config_name,
                                                   self.network_name, self.services_name, self.sim_config_name,
                                                   "best", self.test_mode, f"t_{self.training_id}")
                    self.tb_log_name = os.path.join(self.agent_config_name,
                                                    self.network_name, self.services_name, self.sim_config_name,
                                                    "best", self.test_mode, f"t_{self.training_id}")
                else:
                    # We are in test mode
                    self.result_dir = os.path.join(os.getcwd(), "results", self.agent_config_name,
                                                   self.network_name, self.services_name, self.sim_config_name,
                                                   self.test_mode, f"t_{self.training_id}")
                    self.tb_log_name = os.path.join(self.agent_config_name,
                                                    self.network_name, self.services_name, self.sim_config_name,
                                                    self.test_mode, f"t_{self.training_id}")
                # Change model path
                self.model_path = os.path.join(os.getcwd(), "results", self.agent_config_name,
                                               self.network_name, self.services_name, self.sim_config_name,
                                               self.test_mode, "model.zip")
        else:
            self.result_dir = os.path.join(os.getcwd(), "results", self.agent_config_name,
                                           self.network_name, self.services_name, self.sim_config_name,
                                           self.training_id)
            self.tb_log_name = os.path.join(self.agent_config_name,
                                            self.network_name, self.services_name, self.sim_config_name,
                                            self.training_id)

        os.makedirs(self.result_dir, exist_ok=True)

    def select_best_agent(self):
        agent_result_dir = os.path.join(os.getcwd(), "results", self.agent_config_name,
                                        self.network_name, self.services_name, self.sim_config_name)
        agent_training_ids = os.listdir(agent_result_dir)

        best_ratio = None
        best_delay = None
        best_training_id = None
        for training_id in agent_training_ids:
            # get first subdir of agent dir = first test dir
            test_dirs = next(os.walk(f"{agent_result_dir}/{training_id}"))[1]
            if len(test_dirs) == 0 or training_id == 'best':
                # Not a test_dir
                continue
            # Select a single test directory
            test_dir = test_dirs[0]
            # compare avg testing metrics and choose best
            metrics = pd.read_csv(f"{agent_result_dir}/{training_id}/{test_dir}/metrics.csv")
            # get the last element
            total_flows = float(metrics['total_flows'].tail(1))
            successful_flows = float(metrics['successful_flows'].tail(1))
            avg_e2e_delay = float(metrics['avg_end2end_delay'].tail(1))
            success_ratio = successful_flows / total_flows
            if best_ratio is None or success_ratio >= best_ratio:
                # More successful flows, keep it regardless of delay
                if best_ratio is None or success_ratio > best_ratio:
                    best_ratio = success_ratio
                    best_delay = avg_e2e_delay
                    best_training_id = training_id
                else:
                    # Same successful flows, check flow better delay
                    if avg_e2e_delay < best_delay:
                        best_ratio = success_ratio
                        best_delay = avg_e2e_delay
                        best_training_id = training_id
        return best_training_id
