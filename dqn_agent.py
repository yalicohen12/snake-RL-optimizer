import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from dqn import DQN, ReplayMemory
import cv2


class DQNAgent:
    def __init__(self, observations_dim, actions_dim, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, target_update=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observations_dim = observations_dim
        self.actions_dim = actions_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = DQN(observations_dim, actions_dim).to(self.device)
        self.target_net = DQN(observations_dim, actions_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.actions_dim)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.cat(batch[4])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, env, num_episodes, max_steps_per_episode):
        print("num_episodes: " , num_episodes)
        training_rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            total_reward = 0
            episode_reward = 0

            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, done = env.step(action.item())
                total_reward += reward
                episode_reward += reward

                reward = torch.tensor([reward], device=self.device)
                done = torch.tensor([float(done)], device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, reward, next_state, done)

                state = next_state

                self.optimize_model()
                self.update_epsilon()
                self.update_target_network()

                self.steps_done += 1

                if done:
                    break

            training_rewards.append(episode_reward)

            # print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.2f}")
        return training_rewards

    def evaluate(self, env, num_episodes):
        self.policy_net.eval()  # Set the network to evaluation mode
        evaluation_rewards = []
        won_count = 0

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.policy_net(state).max(1)[1].view(1, 1)
                next_state, reward, done = env.step(action.item())
                episode_reward += reward
                state = next_state

            evaluation_rewards.append(episode_reward)

        self.policy_net.train()  # Set the network back to training mode

        avg_reward = np.mean(evaluation_rewards)
        std_reward = np.std(evaluation_rewards)

        print(f"Evaluation over {num_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Standard Deviation: {std_reward:.2f}")

        return avg_reward, std_reward, evaluation_rewards

    def evaluate_with_video(self, env, num_episodes, video_episodes=[0, 1], max_steps=1000, video_fps=10):
        self.policy_net.eval()
        evaluation_rewards = []
        won_counter = 0

        for eval_episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            # Initialize video writer if this is a video episode
            if eval_episode in video_episodes:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(f'snake_movement_eval_{eval_episode}.avi', fourcc, video_fps,
                                               (1000, 1000))

            while not done and step_count < max_steps:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.policy_net(state_tensor).max(1)[1].view(1, 1).item()
                next_state, reward, done = env.step(action)
                episode_reward += reward
                step_count += 1

                # Record video frame if this is a video episode
                if eval_episode in video_episodes:
                    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
                    head_position = env.snake_head_position.astype(int)
                    if step_count<100:
                        env.info()

                    # Draw snake head
                    cv2.circle(frame, (head_position[0], head_position[1]), 10, (0, 255, 0), -1)

                    # Calculate and draw snake body
                    radians = np.deg2rad(env.snake_head_angle)
                    end_x = int(head_position[0] + 100 * np.cos(radians))
                    end_y = int(head_position[1] + 100 * np.sin(radians))
                    cv2.line(frame, (head_position[0], head_position[1]), (end_x, end_y), (0, 255, 0), 2)

                    # Draw goal
                    goal_position = env.goal_position.astype(int)
                    if 0 <= goal_position[0] < 1000 and 0 <= goal_position[1] < 1000:
                        cv2.circle(frame, (goal_position[0], goal_position[1]), 10, (0, 0, 255), -1)
                    else:
                        print(f"Warning: Goal position out of bounds: {goal_position}")

                    video_writer.write(frame)

                state = next_state

                if done:
                    won_counter += 1

            evaluation_rewards.append(episode_reward)
            print(f'Evaluation Episode {eval_episode + 1}/{num_episodes}, '
                  f'Total Reward: {episode_reward}, Steps Taken: {step_count}')

            # Close video writer if this was a video episode
            if eval_episode in video_episodes:
                video_writer.release()

        self.policy_net.train()

        avg_reward = np.mean(evaluation_rewards)
        std_reward = np.std(evaluation_rewards)

        print(f"Evaluation over {num_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Standard Deviation: {std_reward:.2f}")
        print(f"Games won: {won_counter}/{num_episodes}")

        return avg_reward, std_reward, evaluation_rewards, won_counter

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy_net(state).max(1)[1].view(1, 1).item()