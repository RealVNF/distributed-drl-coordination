import gym
from spr_rl.agent.params import Params
from gym.spaces import Discrete, Box
from gym.utils import seeding
from spr_rl.envs.wrapper import SPRSimWrapper
import numpy as np
import random
import csv


class SprEnv(gym.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc...
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self.params: Params = kwargs['params']
        # Obs space is currently temporary
        self.action_space = Discrete(self.params.action_limit)
        # Upper bound for the obs here is 1000, I assume TTL no higher than 1000
        self.observation_space = Box(-1, 1000, shape=self.params.observation_shape)
        self.wrapper = SPRSimWrapper(self.params)
        if not self.params.test_mode:
            file_hash = random.randint(0, 99999999)
            file_name = f"episode_reward_{file_hash}.csv"
            self.episode_reward_stream = open(f"{self.params.result_dir}/{file_name}", 'a+', newline='')
            self.episode_reward_writer = csv.writer(self.episode_reward_stream)
            self.episode_reward_writer.writerow(['episode', 'reward'])
        self.episode_number = -1  # -1 here so that first episode reset call makes it 0

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Get flow information before action
        processing_index = self.last_flow.processing_index
        forward_to_eg = self.last_flow.forward_to_eg
        previous_node_id = self.last_flow.current_node_id
        flow_delay = self.last_flow.end2end_delay

        # Apply action
        nn_state, sim_state = self.wrapper.apply(action)
        new_flow = sim_state.flow

        sfc_len = len(sim_state.sfcs[self.last_flow.sfc])

        # Set reward points
        SUCCESS = 10
        PROCESSED = 1 / sfc_len
        EG_MOVED = -(self.last_flow.end2end_delay - flow_delay) / self.params.net_diameter
        EG_KEPT = -1 / self.params.net_diameter
        DROPPED = -10
        MOVED = 0

        # This reward works by using the concept of aliasing and tracking the flow object in memory
        if self.last_flow.success:
            # If flow successful
            reward = SUCCESS
        else:
            if self.last_flow.dropped:
                # If the flow was dropped
                reward = DROPPED
            else:
                if forward_to_eg:
                    if self.last_flow.current_node_id == self.last_flow.egress_node_id:
                        # Flow arrived at egress, wont ask for more decisions
                        reward = SUCCESS
                    else:
                        if self.last_flow.current_node_id == previous_node_id:
                            # Flow stayed at the node
                            reward = EG_KEPT
                        else:
                            # Flow moved
                            reward = EG_MOVED
                else:
                    # Flow is still processing
                    # if flow processed more
                    if self.last_flow.processing_index > processing_index:
                        if (
                            self.last_flow.current_node_id == self.last_flow.egress_node_id
                        ) and (
                            self.last_flow.processing_index == sfc_len
                        ):
                            # Flow was processed at last sf at egress node,
                            # but success wont be triggered as it will automatically depart
                            reward = SUCCESS
                        else:
                            reward = PROCESSED
                    else:
                        reward = MOVED

        done = False
        # Episode length is a set number of flows
        self.episode_reward += reward
        if not self.params.test_mode and self.wrapper.simulator.env.now >= self.params.episode_length:
            done = True
            self.episode_reward_writer.writerow([self.episode_number, self.episode_reward])
        self.steps += 1

        # Set last flow to new flow. New actions will be generated for the new flow
        self.last_flow = new_flow
        return nn_state, reward, done, {'sim_time': self.wrapper.simulator.env.now}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        if self.params.sim_seed is None:
            sim_seed = self.random_gen.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
        else:
            sim_seed = self.params.sim_seed
        nn_state, sim_state = self.wrapper.init(sim_seed)

        self.steps = 0
        self.episode_reward = 0
        self.episode_number += 1

        self.last_flow = sim_state.flow
        self.network = sim_state.network

        return nn_state

    @staticmethod
    def get_dist_to_eg(network, flow):
        """ Returns the distance to egress node in hops """
        dist_to_egress = network.graph['shortest_paths'][(flow.current_node_id,
                                                          flow.egress_node_id)][1]  # 1: delay; 2: hops
        return dist_to_egress

    def render(self, mode='cli'):
        assert mode in ['human']

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.random_gen, seed = seeding.np_random()
        return [seed]
