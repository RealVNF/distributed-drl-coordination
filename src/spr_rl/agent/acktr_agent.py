from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, FeedForwardPolicy
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import BaseCallback
from tqdm.auto import tqdm
from .params import Params
from spr_rl.envs.spr_env import SprEnv
import tensorflow as tf
from tensorflow.nn import relu, tanh
import csv
import sys
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Progress bar code from
# https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/4_callbacks_hyperparameter_tuning.ipynb
class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class SPRPolicy(FeedForwardPolicy):
    """
    Custom policy. Exactly the same as MlpPolicy but with different NN configuration
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        self.params: Params = _kwargs['params']
        pi_layers = self.params.agent_config['pi_nn']
        vf_layers = self.params.agent_config['vf_nn']
        activ_function_name = self.params.agent_config['nn_activ']
        activ_function = eval(activ_function_name)
        net_arch = [dict(vf=vf_layers, pi=pi_layers)]
        super(SPRPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        net_arch=net_arch, act_fun=activ_function, feature_extraction="spr", **_kwargs)


class ACKTR_Agent:

    def __init__(self, params: Params):
        self.params: Params = params
        policy_name = self.params.agent_config['policy']
        self.policy = eval(policy_name)

    def create_model(self, n_envs=1):
        """ Create env and agent model """
        env_cls = SprEnv
        self.env = make_vec_env(env_cls, n_envs=n_envs, env_kwargs={"params": self.params}, seed=self.params.seed)
        self.model = ACKTR(
            self.policy,
            self.env,
            gamma=self.params.agent_config['gamma'],
            n_steps=self.params.agent_config['n_steps'],
            ent_coef=self.params.agent_config['ent_coef'],
            vf_coef=self.params.agent_config['vf_coef'],
            vf_fisher_coef=self.params.agent_config['vf_fisher_coef'],
            max_grad_norm=self.params.agent_config['max_grad_norm'],
            learning_rate=self.params.agent_config['learning_rate'],
            gae_lambda=self.params.agent_config['gae_lambda'],
            lr_schedule=self.params.agent_config['lr_schedule'],
            kfac_clip=self.params.agent_config['kfac_clip'],
            kfac_update=self.params.agent_config['kfac_update'],
            async_eigen_decomp=self.params.agent_config['async_eigen_decomp'],
            verbose=self.params.agent_config['verbose'],
            tensorboard_log="./tb/acktr/",
            seed=self.params.seed,
            policy_kwargs={"params": self.params}
        )

    def train(self):
        with ProgressBarManager(self.params.training_duration) as callback:
            self.model.learn(
                total_timesteps=self.params.training_duration,
                tb_log_name=self.params.tb_log_name,
                callback=callback)

    def test(self):
        self.params.test_mode = True
        obs = self.env.reset()
        self.setup_writer()
        episode = 1
        step = 0
        episode_reward = [0.0]
        done = False
        # Test for 1 episode
        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, dones, info = self.env.step(action)
            episode_reward[episode - 1] += reward[0]
            if info[0]['sim_time'] >= self.params.testing_duration:
                done = True
                self.write_reward(episode, episode_reward[episode - 1])
                episode += 1
            sys.stdout.write(
                "\rTesting:" +
                f"Current Simulator Time: {info[0]['sim_time']}. Testing duration: {self.params.testing_duration}")
            sys.stdout.flush()
            step += 1
        print("")

    def save_model(self):
        """ Save the model to a zip archive """
        self.model.save(self.params.model_path)

    def load_model(self, path=None):
        """ Load the model from a zip archive """
        if path is not None:
            self.model = ACKTR.load(path)
        else:
            self.model = ACKTR.load(self.params.model_path)
            # Copy the model to the new directory
            self.model.save(self.params.model_path)

    def setup_writer(self):
        episode_reward_filename = f"{self.params.result_dir}/episode_reward.csv"
        episode_reward_header = ['episode', 'reward']
        self.episode_reward_stream = open(episode_reward_filename, 'a+', newline='')
        self.episode_reward_writer = csv.writer(self.episode_reward_stream)
        self.episode_reward_writer.writerow(episode_reward_header)

    def write_reward(self, episode, reward):
        self.episode_reward_writer.writerow([episode, reward])
