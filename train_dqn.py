import cv2
import gym
import numpy as np
import torch

from dqn_agent import DQNAgent
from snake_environment import SnakesWorldEnv
import matplotlib.pyplot as plt
EPISODES = 1000
MAX_STEPS = 1000
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
total_steps = 0

action_mapping = np.array([-30, -15, -5, 0, 5, 15, 30])

env = SnakesWorldEnv()
agent = DQNAgent(observations_dim=env.observation_space.shape[0], actions_dim=7)
training_rewards = agent.train(env, num_episodes=500, max_steps_per_episode=500)

# Plotting training rewards
plt.figure(figsize=(10, 5))
plt.plot(training_rewards)
plt.title('Training Rewards over Episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

avg_reward, std_reward, eval_rewards, won_count = agent.evaluate_with_video(env, num_episodes=50, video_episodes=[0, 49])



# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer = cv2.VideoWriter('snake_movement.avi', fourcc, 10,(1000, 1000))
# EVAL_EPISODES = 50
# agent.policy_net.eval()
# won_counter = 0
# for eval_episode in range(EVAL_EPISODES):
#     state = env.reset()
#     total_reward = 0
#     done = False
#     step_count = 0
#
#     while not done and step_count< MAX_STEPS:
#         action = agent.model_act(state, step=step_count)
#         next_state, reward, done = env.step(action)
#         state = next_state
#         total_reward += reward
#         step_count += 1
#
#         if eval_episode == 40:
#             if step_count % 20 == 0:
#                 print(env.info())
#             frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
#             head_position = env.snake_head_position.astype(int)
#
#             # Draw snake head
#             cv2.circle(frame, (head_position[0], head_position[1]), 10, (0, 255, 0), -1)
#
#             # Calculate the endpoint of the snake based on its length and angle
#             radians = np.deg2rad(env.snake_head_angle)
#             end_x = int(head_position[0] + 100 * np.cos(radians))
#             end_y = int(head_position[1] + 100 * np.sin(radians))
#
#             # Draw the snake's length as a line
#             cv2.line(frame, (head_position[0], head_position[1]), (end_x, end_y), (0, 255, 0), 2)
#             # Check if goal position is valid and within the frame
#             goal_position = env.goal_position.astype(int)
#             if 0 <= goal_position[0] < 1000 and 0 <= goal_position[1] < 1000:
#                 # Draw the goal position
#                 cv2.circle(frame, (goal_position[0], goal_position[1]), 10, (0, 0, 255), -1)  # Red goal
#             else:
#                 print(f"Warning: Goal position out of bounds: {goal_position}")
#
#             # Write the frame to the video
#             video_writer.write(frame)
#
#         if done:
#             print("game won in step: ", step_count)
#             won_counter += 1
#
#     print(f'Evaluation Episode {eval_episode + 1}/{EVAL_EPISODES}, '
#     f'Total Reward: {total_reward}, Steps Taken: {step_count}')
#
# video_writer.release()
# cv2.destroyAllWindows()
# print("won count is: ", won_counter)