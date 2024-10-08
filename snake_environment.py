import math
import gym
from gym import spaces
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge



class SnakesWorldEnv(gym.Env):
    def __init__(self):
        self.board_size = 1000
        self.cell_size = int(self.board_size / 10)
        self.time_step = 1
        self.max_turn_rate = 30
        self.max_snake_length = 100
        self.squint_angle = 45
        self.arrived_distance = 40
        self.view_distance = 300
        self.movement_speed = 50
        self.steps = 0
        self.max_steps = 1000

        self.is_target_visible = False
        self.angle_to_target = -1
        self.snake_head_position = np.zeros(2)
        self.snake_head_angle = 0
        self.goal_position = np.zeros(2)

        self.grid_size = self.board_size // self.cell_size
        self.visited_cells_map = np.zeros((self.grid_size, self.grid_size))
        self.current_seen_cells = 0
        self.current_unseen_cells = 0

        self.distance_to_goal = 1
        self.near_border_counter = 0


        self.observation_space = spaces.Box(
            low=np.array([0, 0, -180, 0, 0, -self.max_turn_rate, 0, 0]),
            high=np.array([self.board_size, self.board_size, 180, self.board_size, self.board_size, self.max_turn_rate, 0,
                           np.linalg.norm(np.array([self.board_size, self.board_size]))]),
            dtype=np.float32
        )
        np.set_printoptions(suppress=True, precision=6)

        self.action_space = spaces.Discrete(7)
        self.actions = np.array([-30, -15, -5, 0, 5, 15, 30])

    def reset(self, *, seed=None, options=None):
        self.snake_head_position = np.random.uniform(50, self.board_size - 50, size=(2,))
        # self.snake_head_position[0] = 500
        # self.snake_head_position[1] = 800
        self.snake_head_angle = np.random.uniform(-180, 180)
        # self.snake_head_angle = 90
        self.goal_position = np.random.uniform(50, self.board_size - 50, size=(2,))
        # self.goal_position[0] = 443
        # self.goal_position[1] = 709
        self.distance_to_goal = 1
        self.visited_cells_map = np.zeros((self.grid_size, self.grid_size))
        self.is_target_visible = False
        self.current_seen_cells = 0
        self.current_unseen_cells = 0
        self.near_border_counter = 0
        self.steps = 0
        self.max_steps = 1000
        self.angle_to_target = -1
        return self._get_obs()

    def reset_goal(self):
        self.goal_position = np.random.uniform(50, self.board_size - 50, size=(2,))
        self.distance_to_goal = 1
        self.visited_cells_map = np.zeros((self.grid_size, self.grid_size))
        self.is_target_visible = False
        self.current_seen_cells = 0
        self.current_unseen_cells = 0
        self.near_border_counter = 0
        self.steps = 0
        self.max_steps = 1000
        self.angle_to_target = -1
        return self._get_obs()


    def _get_obs(self):
        return self.normalize_obs()

    def step(self, action):
        self.steps += 1
        reward = 0
        done = False

        self.snake_head_angle = self.get_updated_snake_head_angle(self.actions[action])
        new_x, new_y = self.get_updated_snake_position()
        if self.angle_to_target != -1:
            self.angle_to_target = self.get_current_angle_to_target()

        if self.is_near_border(new_x, new_y):
            self.near_border_counter += 1
            self.snake_head_angle *= 1.15
            reward -= self.scale_border_visit_counter() * 2
        else:
            if self.near_border_counter >0:
                reward += 0.3
            self.near_border_counter = 0

        old_position = self.snake_head_position.copy()
        self.snake_head_position[0] = new_x
        self.snake_head_position[1] = new_y

        movement = np.linalg.norm(self.snake_head_position - old_position)
        reward += (movement / self.board_size) * 2

        seen_cells, unseen_cells = self.get_cells_status_within_fov()
        self.current_seen_cells = seen_cells
        self.current_unseen_cells = unseen_cells

        reward += 1.5 * unseen_cells / (self.grid_size ** 2)

        reward -= seen_cells / (self.grid_size ** 2)

        if seen_cells and unseen_cells ==0:
            reward -= 0.2

        if self.is_in_fov(self.goal_position):
            self.is_target_visible = True
            self.angle_to_target = self.get_current_angle_to_target()
            if np.linalg.norm(self.snake_head_position - self.goal_position) <= self.arrived_distance:
                reward += 100
                done = True
            else:
                reward += self.evaluate_reward_inside_fov()
                self.distance_to_goal = self.get_updated_distance_to_goal()
        else:
            if self.is_target_visible:
                reward -= 0.25
            self.is_target_visible = False
            reward += self.evaluate_reward_outside_fov()

            if self.distance_to_goal != 1:
                new_distance_to_goal = self.get_updated_distance_to_goal()
                reward += (new_distance_to_goal - self.distance_to_goal) * 5
                self.distance_to_goal = new_distance_to_goal

        observation = self._get_obs()
        return observation, reward, done

    def is_in_fov(self, goal_position):
        dx = goal_position[0] - self.snake_head_position[0]
        dy = goal_position[1] - self.snake_head_position[1]
        angle_to_goal = self.get_current_angle_to_target()
        angle_diff = (angle_to_goal - self.snake_head_angle + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        distance_to_goal = np.sqrt(dx ** 2 + dy ** 2)
        return abs(angle_diff) <= self.squint_angle and distance_to_goal <= self.view_distance

    def get_updated_snake_head_angle(self, action):
        return ((self.snake_head_angle + action + 180) % 360) - 180

    def get_updated_snake_position(self):
        radians = np.deg2rad(self.snake_head_angle)
        delta_x = self.time_step * self.movement_speed * np.cos(radians)
        delta_y = self.time_step * self.movement_speed * np.sin(radians)

        new_x = np.clip(self.snake_head_position[0] + delta_x, 0, self.board_size)
        new_y = np.clip(self.snake_head_position[1] + delta_y, 0, self.board_size)

        return new_x, new_y

    def get_current_angle_to_target(self):
        dx = self.goal_position[0] - self.snake_head_position[0]
        dy = self.goal_position[1] - self.snake_head_position[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return (angle + 180) % 360 - 180

    def get_angle_difference_to_target(self):
        current_angle = self.get_current_angle_to_target()
        angle_diff = (current_angle - self.snake_head_angle + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        return abs(angle_diff)

    def is_near_border(self, x, y):
        # print("working with: ",x,y, " ange: ", self.snake_head_angle)
        border_distance = 100
        borders = [
            (x, self.board_size, 90),
            (x, 0, -90),
            (0, y, 180),
            (self.board_size, y, 0)
        ]
        for border_x, border_y, border_angle in borders:
            if self.is_snake_seeing_border((border_x, border_y), border_angle, border_distance):
                # print("returning true")
                return True
        # print("returning false")
        return False

    def is_point_in_fov(self, point, max_distance):
        dx = point[0] - self.snake_head_position[0]
        dy = point[1] - self.snake_head_position[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance > max_distance:
            return False
        angle_diff = self.get_relative_angle(point)
        return abs(angle_diff) <= self.squint_angle

    def is_snake_seeing_border(self,border_point,border_angle,border_distance):
        dx = border_point[0] - self.snake_head_position[0]
        dy = border_point[1] - self.snake_head_position[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance > border_distance:
            return False

        angle_to_border = math.degrees(math.atan2(dy, dx))

        angle_to_border = (angle_to_border + 180) % 360 - 180

        snake_angle = self.snake_head_angle
        angle_diff_to_snake = min(abs(angle_to_border - snake_angle), 360 - abs(angle_to_border - snake_angle))

        if angle_diff_to_snake <= self.squint_angle + 30:
            projection = abs(dx * math.cos(math.radians(snake_angle)) + dy * math.sin(math.radians(snake_angle)))
            if projection < border_distance * math.sin(math.radians(self.squint_angle)):
                return True

        return False

    def get_relative_angle(self, point):
        dx = point[0] - self.snake_head_position[0]
        dy = point[1] - self.snake_head_position[1]
        angle_to_point = np.degrees(np.arctan2(dy, dx)) % 360
        relative_angle = (angle_to_point - self.snake_head_angle + 180) % 360 - 180
        return relative_angle

    def evaluate_reward_outside_fov(self):
        reward = -0.1
        if self.angle_to_target != -1:
            angle_diff = self.get_angle_difference_to_target()
            # print("angle diff: ", angle_diff)
            if angle_diff < self.squint_angle:
                # print(f"Updated reward after angle alignment: {reward}")
                reward += (self.squint_angle - angle_diff) / self.squint_angle
            else:
                reward -= angle_diff / 180
        return reward

    def evaluate_reward_inside_fov(self):
        curr_distance = self.distance_to_goal
        new_distance_to_goal = self.get_updated_distance_to_goal()
        # print("inside with: " , curr_distance , new_distance_to_goal)
        if curr_distance !=1:
            return (curr_distance - new_distance_to_goal) * 10
        return (curr_distance - new_distance_to_goal) / 1.2

    def get_cells_status_within_fov(self):
        count_seen_cells = 0
        count_unseen_cells = 0
        for cell_x in range(self.grid_size):
            for cell_y in range(self.grid_size):
                cell_center = ((cell_x + 0.5) * self.cell_size, (cell_y + 0.5) * self.cell_size)
                if self.is_point_in_fov(cell_center, self.view_distance):
                    if self.visited_cells_map[cell_x, cell_y] == 0:
                        count_unseen_cells += 1
                        self.visited_cells_map[cell_x, cell_y] = 1
                    else:
                        count_seen_cells += 1
                        self.visited_cells_map[cell_x, cell_y] += 1
        return count_seen_cells, count_unseen_cells

    def scale_cells_count(self, count):
        return count / (self.grid_size ** 2)

    def get_updated_distance_to_goal(self):
        dx = self.goal_position[0] - self.snake_head_position[0]
        dy = self.goal_position[1] - self.snake_head_position[1]
        return np.sqrt(dx ** 2 + dy ** 2) / self.board_size

    def normalize_obs(self):
        distance_r, distance_l, distance_u, distance_d = self.normalized_distance_to_borders()
        scaled_border_count = self.scale_border_visit_counter()
        proximity_to_border = min(distance_r, distance_l, distance_u, distance_d)

        return np.array([
            self.normalize(self.snake_head_position[0], 0, self.board_size),
            self.normalize(self.snake_head_position[1], 0, self.board_size),
            self.normalize_angle(self.snake_head_angle),
            # self.scale_cells_count(self.current_seen_cells),
            # self.scale_cells_count(self.current_unseen_cells),
            self.distance_to_goal,
            self.normalize_angle(self.angle_to_target),
            scaled_border_count,
            1 if self.is_target_visible else 0,
            proximity_to_border
        ])

    def distance_change_to_target(self):
        curr_distance = self.distance_to_goal
        new_distance = np.linalg.norm(self.snake_head_position - self.goal_position) / self.board_size
        self.distance_to_goal = new_distance
        return new_distance - curr_distance

    def normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def normalize_angle(self, angle):
        if angle == -1:
            return -1
        # print("angle is: " , angle/180)
        return angle / 180

    def normalize_angle_to_180(self,angle):
        return (angle + 180) % 360 - 180

    def normalized_distance_to_borders(self):
        x, y = self.snake_head_position
        return [
            x / self.board_size,
            (self.board_size - x) / self.board_size,
            y / self.board_size,
            (self.board_size - y) / self.board_size
        ]

    def scale_border_visit_counter(self):
        return min(self.near_border_counter / 50, 10)

    def info(self):
        obs = self._get_obs()
        obs_dict = {
            "Snake Head X": obs[0],
            "Snake Head Y": obs[1],
            "Snake Head Angle": obs[2],
            "Seen Cells (scaled)": obs[3],
            "Unseen Cells (scaled)": obs[4],
            "Distance to Goal": obs[5],
            "angle to target": obs[6],
            "Distance to Right Border": obs[7],
            "Distance to Left Border": obs[8],
            "Distance to Upper Border": obs[9],
            "Distance to Lower Border": obs[10],
            "Border Visit Counter (scaled)": obs[11],
            "Steps Taken (normalized)": obs[13],
            "is target visible: ": obs[12],
        }

        print("Observation Information:")
        for key, value in obs_dict.items():
            print(f"{key}: {value:.6f}")

    def plot(self, obs):
        fig, ax = plt.subplots(figsize=(10, 6))  # Make the figure wider
        ax.set_xlim(0, self.board_size)
        ax.set_ylim(0, self.board_size)

        # Plot the grid
        for x in range(0, self.board_size, self.cell_size):
            for y in range(0, self.board_size, self.cell_size):
                ax.add_patch(plt.Rectangle((x, y), self.cell_size, self.cell_size, fill=False, color='gray', lw=0.5))

        # Plot the snake's head position
        snake_x, snake_y = obs[0] * self.board_size, obs[1] * self.board_size
        ax.plot(snake_x, snake_y, 'bo', label='Snake Head', markersize=10)

        # Plot the snake's direction as an arrow
        angle_rad = np.deg2rad(obs[2] * 180)
        arrow_length = 50
        ax.arrow(snake_x, snake_y, arrow_length * np.cos(angle_rad), arrow_length * np.sin(angle_rad),
                 head_width=10, head_length=20, fc='blue', ec='blue', label='Snake Direction')

        # Plot the goal position
        goal_x, goal_y = self.goal_position[0], self.goal_position[1]
        ax.plot(goal_x, goal_y, 'ro', label='Goal', markersize=10)

        # Mark visited cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.visited_cells_map[i, j] > 0:
                    cell_x = i * self.cell_size
                    cell_y = j * self.cell_size
                    ax.add_patch(plt.Rectangle((cell_x, cell_y), self.cell_size, self.cell_size,
                                               color='green', alpha=0.3))

        # Plot the snake's field of view (FOV)
        fov_angle_min = obs[2] * 180 - self.squint_angle
        fov_angle_max = obs[2] * 180 + self.squint_angle
        wedge = Wedge((snake_x, snake_y), self.view_distance, fov_angle_min, fov_angle_max,
                      color='yellow', alpha=0.2, label='FOV')
        ax.add_patch(wedge)

        # Add parameters to the side
        # seen_cells = obs[3]
        # unseen_cells = obs[4]
        snake_head_pos = [obs[0], obs[1]]
        angle_value = obs[2] * 180
        distance_to_goal = obs[3]
        angle = obs[6]
        border_vis = obs[5] * 100
        is_target_visible = obs[6]
        prox = obs[7]

        side_info = (
            # f"Seen Cells: {seen_cells:.3f}\n"
            # f"Unseen Cells: {unseen_cells:.3f}\n"
            f"snake_head pos: {snake_head_pos} \n"
            f"Angle: {angle_value:.2f}°\n"
            f"Distance to Goal: {distance_to_goal:.3f}\n"
            f"angle to traget: {angle:.3f}\n"
            f"is target visible: {is_target_visible} \n"
            f" proximity to border: {prox} \n"
            f"border visit count: {border_vis}"

        )

        # Adjust text placement to avoid cutting
        ax.text(self.board_size * 1.1, self.board_size * 0.8, side_info,
                fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        # Extend the x-limits to make room for text
        ax.set_xlim(0, self.board_size * 1.4)

        # Add legend and labels
        ax.legend()
        ax.set_title('Snake World Environment')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(False)
        plt.tight_layout()  # Ensure the layout fits well
        plt.show()

#
# env = SnakesWorldEnv()
# obs = env.reset()
# # env.info()
# env.plot(obs)
#
# obs1, r,_ = env.step(3)
# # env.info()
# print("reward for the first step: ",r)
# env.plot(obs1)
#
# obs2,r,_ = env.step(3)
# # env.info()
# print("reward for the second step: ",r)
# env.plot(obs2)
#
# obs3, r,_ = env.step(6)
# print("reward for the third step: ",r)
# env.plot(obs3)
# # env.info()
# obs4 ,r,_ = env.step(1)
# print("reward for the 4th step: ",r)
# # env.info()
# env.plot(obs4)
# obs5 ,r, _ = env.step(4)
# print("reward for the 5th step: ",r)
# # env.info()
# env.plot(obs5)
#
# obs6, r, _ = env.step(4)
# print("reward for the 6th step: ",r)
# # env.info()
# env.plot(obs6)
#
# obs7 , r, _ = env.step(6)
# print("reward for the 7th step: ",r)
# # env.info()
# env.plot(obs7)
