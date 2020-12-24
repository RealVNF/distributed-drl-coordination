import click
import random
from spr_rl.agent import ACKTR_Agent
from spr_rl.agent import Params
import os
import inspect
import numpy as np
from spr_rl.envs.spr_env import SprEnv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Click decorators
# TODO: Add testing flag (timestamp and seed). Already set in params
@click.command()
@click.argument('network', type=click.Path(exists=True))
@click.argument('agent_config', type=click.Path(exists=True))
@click.argument('simulator_config', type=click.Path(exists=True))
@click.argument('services', type=click.Path(exists=True))
@click.argument('training_duration', type=int)
@click.option('-s', '--seed', type=int, help="Set the agent's seed", default=None)
@click.option('-t', '--test', help="Path to test timestamp and seed", default=None)
@click.option('-a', '--append_test', help="test after training", is_flag=True)
@click.option('-m', '--model_path', help="path to a model zip file", default=None)
@click.option('-ss', '--sim-seed', type=int, help="simulator seed", default=None)
@click.option('-b', '--best', help="Select the best agent", is_flag=True)
def main(network, agent_config, simulator_config, services, training_duration,
         seed, test, append_test, model_path, sim_seed, best):
    """
    SPR-RL DRL Scaling and Placement main executable
    """
    # Get or set a seed
    if seed is None:
        seed = random.randint(0, 9999)

    # Seed random generators
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"Creating agent with seed: {seed}")
    print(f"Using network: {network}")

    if best:
        # Just a placeholder to trick params into thinking it is in test mode
        test = "placeholder"

    # Create the parameters object
    params = Params(
        seed,
        agent_config,
        simulator_config,
        network,
        services,
        training_duration=training_duration,
        test_mode=test,
        sim_seed=sim_seed,
        best=best
    )

    # Create the agent
    agent_class = eval(params.agent_config['agent'])
    agent = agent_class(params)

    if test is None:
        # Create model and environment
        agent.create_model(n_envs=params.agent_config['n_env'])
        # Train the agent
        print(f"Training for {training_duration} steps.")
        agent.train()

        # Save the model
        print(f"Saving the model to {params.model_path}")
        print(f"Agent training ID: {params.training_id}")
        agent.save_model()

    # Check to see if testing or append test is set to test the agent
    if test is not None or append_test:
        if append_test:
            params.test_mode = True
            params.create_result_dir()
        # Create or recreate model
        agent.create_model()
        # Load weigths
        agent.load_model(model_path)
        print("Testing for 1 episode")
        agent.test()
    print("Storing reward function")
    store_reward_function(params)
    print("Done")


def store_reward_function(params: Params):
    reward_path = os.path.join(params.result_dir, "reward.py")
    with open(reward_path, 'w') as reward_file:
        reward_function = inspect.getsource(SprEnv.step)
        reward_file.write(reward_function)


if __name__ == "__main__":
    agent_config = "inputs/config/drl/acktr/acktr_default_4-env.yaml"
    network = "inputs/networks/interroute-in2-eg1-rand-cap0-2.graphml"
    services = "inputs/services/abc-start_delay0.yaml"
    sim_config = "inputs/config/simulator/mmpp-12-8.yaml"
    training_duration = "200000"
    main([network, agent_config, sim_config, services, training_duration, '-a', '-s', '8443'])
    # main([network, agent_config, sim_config, services, training_duration, '-t', '2020-12-03_13:17:26_seed9834'])

    # main([network, agent_config, sim_config, services, training_duration, '--best'])
    # main([network, agent_config, sim_config, services, training_duration, '-t', 'best',
    #       '-m', 'results/models/poisson/model.zip'])
