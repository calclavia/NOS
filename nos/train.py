import gym
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

import meta_env
from . import algo
from .policy import CustomLSTMPolicy

meta_env.register()

batch_size = 5

env = DummyVecEnv([lambda: gym.make('MetaEnv-v0', cuda=True) for _ in range(batch_size)])  # The algorithms require a vectorized environment to run
env.get_valid_actions = lambda: np.array([e.get_valid_actions() for e in env.envs])

model = algo.MaskedPPO(CustomLSTMPolicy, env, verbose=1,
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