import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback

# env = gym.make('PathFollowingEnv-v0')
# # env = gym.make('CartPole-v1')
# obs = env.reset()
# # for _ in range(5):
# #     action_sample = env.action_space.sample()
# for _ in range(1):
#     env.eval(env.action_space.sample()) # take a random action
#     # env.step([10e5])

# env.close()

from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import DQN

# model = SAC('MlpPolicy', 'PathFollowingEnv-v0', verbose=1, tensorboard_log="./sac_path_tensorboard/")
model = DQN('MlpPolicy', 'PathFollowingEnv-v0', verbose=1, tensorboard_log="./dqn_path_tensorboard/")
# model = DQN('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./dqn_cartpole_tensorboard/")
model.learn(total_timesteps=100000)
model.save("dqn_path_100000")
env = gym.make('PathFollowingEnv-v0')
obs = env.reset()
action, _ = model.predict(obs)
env.eval(action)
print(action)