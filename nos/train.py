import gym
import numpy as np
import torch
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

import meta_env
from . import algo
from .policy import CustomLSTMPolicy

meta_env.register()

if __name__ == "__main__":
    batch_size = 5
    num_envs = 5
    num_gpus = torch.cuda.device_count()

    def make_env(index):
        return lambda: gym.make('MetaEnv-v0', device=torch.device('cuda', index=index % num_gpus))
    
    env = SubprocVecEnv([make_env(x) for x in range(num_envs)], start_method='forkserver')

    # env.get_valid_actions = lambda: np.array([e.get_valid_actions() for e in env.envs])
    env.get_valid_actions = lambda: np.array(env.env_method('get_valid_actions'))

    model = algo.MaskedPPO(CustomLSTMPolicy, env, verbose=1, n_steps=5, ent_coef=0.0015,
        nminibatches=batch_size, learning_rate=1e-5, tensorboard_log="../out/meta_opt/")

    model.learn(total_timesteps=100000, log_interval=10)
    model.save('meta_optimizer')

    obs = env.reset()
    state = None
    total_rewards = 0
    done = [False for _ in range(env.num_envs)]

    for i in range(1000):
        action, _states = model.predict(obs, state=state, mask=done)
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards

        # if done:
        #     break
    print(total_rewards)