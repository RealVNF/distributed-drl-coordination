from gym.envs.registration import register

register(
    id='SprEnv-v0',
    entry_point='spr_rl.envs:SprEnv',
)
