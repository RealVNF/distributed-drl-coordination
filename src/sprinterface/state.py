from coordsim.network.flow import Flow


class SPRState:
    def __init__(self, flow: Flow, network: dict, sfcs: dict, network_stats: dict):
        """
        SPRState Class for SPR algorithm
        """
        self.flow = flow
        self.network = network
        self.sfcs = sfcs
        self.network_stats = network_stats
