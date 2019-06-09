from .meta_env import MetaEnv
import gym

def register():
    env_name = 'MetaEnv-v0'
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

    gym.envs.register(
        id=env_name,
        entry_point=MetaEnv,
         max_episode_steps=100,
    )
