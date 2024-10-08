import cv2
import gym
import numpy as np
import torch
from dqn_agent import DQNAgent
from snake_environment import SnakesWorldEnv
import matplotlib.pyplot as plt

# Initialize environment and agent
env = SnakesWorldEnv()
agent = DQNAgent(observations_dim=env.observation_space.shape[0], actions_dim=7)

# Load the saved model weights
agent.policy_net.load_state_dict(torch.load("dqn_snake_model.pth"))
agent.policy_net.eval()  # Set the model to evaluation mode

print("Model loaded successfully!")

# Evaluate the loaded model
# avg_reward, std_reward, eval_rewards, won_count = agent.evaluate_with_video(env, num_episodes=100, video_episodes=[0, 49, 59])

s = agent.evaluate_multiple_targets(env)

# print(f"Average Reward: {avg_reward}")
# print(f"Standard Deviation of Rewards: {std_reward}")
# print(f"Games Won: {won_count}")
#
# # Optionally, you can visualize the rewards
# plt.figure(figsize=(10, 5))
# plt.plot(eval_rewards)
# plt.title('Evaluation Rewards over Episodes')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.show()
