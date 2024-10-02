import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class SnakesWorldEnv(gym.Env):
    def __init__(self):
        self.board_size = 1000
        self.time_step = 0.5
        self.max_turn_rate = 30
        self.max_snake_length = 100
        self.squint_angle = 45

        self.snake_head_position = np.zeros(2)
        self.snake_head_angle = 0
        self.goal_position = np.zeros(2)
        self.distance_to_goal = 0
        self.arrived_distance = 5
        self.previous_action = None
        self.last_seen_target_direction = None
        self.is_target_on_fov = False

        #  observation_space is mostly gpt generated
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -180, 0, 0, -self.max_turn_rate, 0, -180, 0]),
            high=np.array([self.board_size, self.board_size, 180, self.board_size, self.board_size, self.max_turn_rate,
                           np.linalg.norm(np.array([self.board_size, self.board_size])), 180, 1]),
            dtype=np.float32
        )

        self.action_space = (spaces.Box
        (low=np.array([-self.max_turn_rate]),
         high=np.array([self.max_turn_rate]),
         dtype=np.float32))

    def reset(self, *, seed=None, options=None):
        self.snake_head_position = np.random.uniform(0, self.board_size, size=(2,))
        self.snake_head_angle = np.random.uniform(-180, 180)

        self.goal_position = np.random.uniform(0, self.board_size, size=(2,))

        self.distance_to_goal = np.linalg.norm(self.snake_head_position - self.goal_position)

        obs = self._get_obs()
        info = self._get_info()
        return obs , info

    def _get_obs(self):
        distance_to_goal = 0
        if self.is_target_on_fov:
            distance_to_goal = np.linalg.norm(self.snake_head_position - self.goal_position)

        obs = np.array([
            self.snake_head_position[0],
            self.snake_head_position[1],
            self.snake_head_angle,
            self.goal_position[0] if self.is_target_on_fov else 0,
            self.goal_position[1] if self.is_target_on_fov else 0,
            self.previous_action if self.previous_action is not None else 0,
            distance_to_goal,
            self.last_seen_target_direction if self.last_seen_target_direction is not None else 0,
            1 if self.is_target_on_fov else 0
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        return np.linalg.norm(self.snake_head_position - self.goal_position)

    def step(self, action):
        reward = 0
        done = False
        action = np.clip(action, -self.max_turn_rate, self.max_turn_rate)
        self.snake_head_angle = (self.snake_head_angle + action) % 360

        new_x , new_y = self.get_updated_snake_position(action)

        if new_x < 0 or new_x >= self.board_size or new_y < 0 or new_y >= self.board_size:
            reward = -80
            return

        else:
            self.snake_head_position[0] = new_x
            self.snake_head_position[1] = new_y

        if self.is_in_fov(self.goal_position):
           reward = self.evaluate_reward_inside_fov(action)
           self.last_seen_target_direction = self.snake_head_angle
           if reward==100:
               done = True
        else:
            reward = self.evaluate_reward_outside_fov(action)

        success = done

        observation = np.array([
            self.snake_head_position[0], self.snake_head_position[1],
            self.snake_head_angle,
            self.goal_position[0], self.goal_position[1]
        ], dtype=np.float32)

        return  observation, reward, done, success, {}

    #mostly gpt generated
    def is_in_fov(self, goal_position):
        dx = goal_position[0] - self.snake_head_position[0]
        dy = goal_position[1] - self.snake_head_position[1]

        angle_to_goal = np.degrees(np.arctan2(dy, dx)) % 360

        angle_diff = (angle_to_goal - self.snake_head_angle + 360) % 360

        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff <= self.squint_angle

    def get_updated_snake_position(self, action):
        if action > self.max_turn_rate or action < -self.max_turn_rate:
            return self.snake_head_position[0], self.snake_head_position[1]

        #compution here is mostly gpt generated
        self.snake_head_angle = (self.snake_head_angle + action) % 360
        radians = np.deg2rad(self.snake_head_angle)
        delta_x = self.time_step * 100 * np.cos(radians)
        delta_y = self.time_step * 100 * np.sin(radians)

        new_x = self.snake_head_position[0] + delta_x
        new_y = self.snake_head_position[1] + delta_y

        new_x = new_x % self.board_size
        new_y = new_y % self.board_size

        return new_x, new_y


    def evaluate_reward_outside_fov(self, action):
        sharp_turn_threshold = self.max_turn_rate / 2
        gentle_turn_threshold = self.max_turn_rate / 4

        reward = 0

        if self.is_target_on_fov:
            self.is_target_on_fov = False
            reward -= 10

        elif self.previous_action is not None and abs(action - self.previous_action) >= sharp_turn_threshold:
            reward -= 8

        elif abs(action) >= sharp_turn_threshold:
            reward -= 5

        elif abs(action) >= gentle_turn_threshold:
            reward += 2

        if action == 0:
            reward += 1

        if self.last_seen_target_direction is not None:
            relative_angle = self.get_relative_angle_to_target(self.snake_head_angle)
            if abs(relative_angle) <= self.squint_angle:
                reward += 6
            else:
                reward -= 6

        self.previous_action = action
        return reward

    def evaluate_reward_inside_fov(self, action):
        reward = 0
        if not self.is_target_on_fov:
            self.is_target_on_fov = True
            reward += 8

        else:
            new_distance_to_goal = np.linalg.norm(self.snake_head_position - self.goal_position)

            if new_distance_to_goal <= self.arrived_distance:
                reward += 100
                return reward

            reward += abs((self.distance_to_goal - new_distance_to_goal) / self.distance_to_goal) * 50
            self.distance_to_goal = new_distance_to_goal

        self.previous_action = action
        return reward

    def get_relative_angle_to_target(self, snake_heading):
        relative_angle = self.last_seen_target_direction - snake_heading
        relative_angle = (relative_angle + 180) % 360 - 180
        return relative_angle

    def render(self, mode='human'):
        plt.clf()
        plt.xlim(0, self.board_size)
        plt.ylim(0, self.board_size)
        plt.scatter(*self.snake_head_position, color='green', label='Snake Head')
        plt.scatter(*self.goal_position, color='red', label='Goal')
        plt.legend()
        plt.draw()
        plt.pause(0.01)

