from siminterface import Simulator
from spr_rl.agent.params import Params
import numpy as np
from sprinterface.action import SPRAction
from sprinterface.state import SPRState


class SPRSimWrapper:
    def __init__(self, params: Params):
        self.params = params
        # Create the simulator
        self.simulator = Simulator(
            params.network_path,
            params.services_path,
            params.sim_config_path,
            test_mode=params.test_mode,
            test_dir=params.result_dir
        )

        # Placeholder for flow that is being passed from Simulator to agent
        self.flow = None

    def init(self, sim_seed):
        """ Start the simulator and get init state """
        # Generate new seed to seed the simulator
        sim_state: SPRState = self.simulator.init(sim_seed)
        nn_state = self.process_state(sim_state)

        return nn_state, sim_state

    def apply(self, action):
        if action >= len(self.node_and_neighbors):
            destination = None
        else:
            destination = self.node_and_neighbors[action]
        sim_action = SPRAction(self.flow, destination)
        sim_state: SPRState = self.simulator.apply(sim_action)
        nn_state = self.process_state(sim_state)

        return nn_state, sim_state

    def process_state(self, sim_state: SPRState):
        self.flow = sim_state.flow
        self.sfcs = sim_state.sfcs
        self.network = sim_state.network

        flow_proc_percentage = self.flow.current_position / len(self.sfcs[self.flow.sfc])

        # get neighbor nodes
        neighbor_node_ids = list(sim_state.network[self.flow.current_node_id].keys())

        self.node_and_neighbors = [self.flow.current_node_id]

        self.node_and_neighbors.extend([node_id for node_id in neighbor_node_ids])

        remaining_node_resources = np.full((self.params.node_resources_size, ), -1.0, dtype=np.float32)
        for i, node_id in enumerate(self.node_and_neighbors):
            node_remaining_cap = sim_state.network.nodes[node_id]['remaining_cap']

            current_sf = self.flow.current_sf
            if self.flow.forward_to_eg:
                current_sf = self.sfcs[self.flow.sfc][-1]
            resource_function = self.simulator.params.sf_list[current_sf]['resource_function']
            if not self.flow.forward_to_eg:
                requested_resources = resource_function(self.flow.dr)
            else:
                requested_resources = 0
            node_remaining_cap_norm = (node_remaining_cap - requested_resources) / self.params.max_node_cap

            node_remaining_cap_norm = np.clip(node_remaining_cap_norm, -1.0, 1.0)
            # if node_remaining_cap / self.flow.dr >= 1:
            #     node_remaining_cap_norm = 1
            # else:
            #     node_remaining_cap_norm = 0
            # node_remaining_cap_norm = node_remaining_cap
            remaining_node_resources[i] = node_remaining_cap_norm

        remaining_link_resources = np.full((self.params.link_resources_size, ), -1.0, dtype=np.float32)
        for i, node_id in enumerate(neighbor_node_ids):
            link_remaining_cap = sim_state.network[self.flow.current_node_id][node_id]['remaining_cap']

            link_remaining_cap_norm = (link_remaining_cap - self.flow.dr) / self.params.max_link_caps[
                self.flow.current_node_id]
            link_remaining_cap_norm = np.clip(link_remaining_cap_norm, -1.0, 1.0)

            remaining_link_resources[i] = link_remaining_cap_norm

        # If neighbor does not exist, set distance to -1
        neighbors_dist_to_eg = np.full((self.params.neighbor_dist_to_eg, ), -1.0, dtype=np.float32)
        for i, node_id in enumerate(neighbor_node_ids):
            if self.flow.egress_node_id is not None:
                # Check whether distance from current node to neighbor node should also be included
                dist_to_node = self.network.graph['shortest_paths'][(self.flow.current_node_id,
                                                                     node_id)][1]
                dist_to_eg = dist_to_node + self.network.graph['shortest_paths'][(
                    node_id,
                    self.flow.egress_node_id)][1]
                neighbors_dist_to_eg[i] = (self.flow.ttl - dist_to_eg) / self.flow.ttl
            else:
                neighbors_dist_to_eg[i] = -1

        # Component availability status
        vnf_availability = np.full((self.params.vnf_status, ), -1.0, dtype=np.float32)
        for i, node_id in enumerate(self.node_and_neighbors):
            flow_sf = self.flow.current_sf
            if flow_sf in sim_state.network.nodes[node_id]['available_sf']:
                vnf_availability[i] = 1
            else:
                vnf_availability[i] = 0

        # Distance to egress
        # Check if flow has egress node set
        # if self.flow.egress_node_id is not None:
        #     if self.flow.current_node_id == self.flow.egress_node_id:
        #         at_egress = 1
        #     else:
        #         at_egress = 0
            # dist_to_egress = self.network.graph['shortest_paths'][(self.flow.current_node_id,
        #                                                           self.flow.egress_node_id)][1]  # 1: delay; 2: hops
        # else:
        #     # Any node can be egress
        #     dist_to_egress = 0

        # Remaining flow TTL %
        ttl = self.flow.ttl / self.flow.original_ttl
        # Flow DR
        # dr = self.flow.dr
        # Create the NN state vector
        nn_state = np.concatenate(
            (
                flow_proc_percentage,
                ttl,
                # dr,
                # at_egress,
                vnf_availability,
                remaining_node_resources,
                remaining_link_resources,
                neighbors_dist_to_eg
            ),
            axis=None
        )

        return nn_state
