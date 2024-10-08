import math

import numpy as np

# Constants for the board and FOV
BOARD_SIZE = 1000  # Board dimensions (1000 x 1000 units)
VIEW_DISTANCE = 400  # Maximum view distance (in units)
FOV_ANGLE = 90  # Total FOV angle (45 degrees to each side)
CELL_SIZE = 100  # Size of each cell in units (100x100 cells)

# Compute grid size (number of cells in each dimension)
GRID_SIZE = BOARD_SIZE // CELL_SIZE
visited_map = np.zeros((GRID_SIZE, GRID_SIZE))

# Utility function to normalize an angle between 0 and 360 degrees
def normalize_angle(angle):
    return angle % 360


# Function to compute if a cell is within the FOV of the agent
def get_cells_in_fov(agent_x, agent_y, agent_angle):
    visible_cells = []
    count_seen_cells = 0
    count_unseen_cells = 0

    for cell_x in range(GRID_SIZE):
        for cell_y in range(GRID_SIZE):
            # Calculate the center of the current cell
            cell_center_x = (cell_x + 0.5) * CELL_SIZE
            cell_center_y = (cell_y + 0.5) * CELL_SIZE

            # Calculate the distance from the agent to the cell center
            dx = cell_center_x - agent_x
            dy = cell_center_y - agent_y
            distance_to_cell = math.sqrt(dx ** 2 + dy ** 2)

            # Skip if the cell is beyond the view distance
            if distance_to_cell > VIEW_DISTANCE:
                continue

            # Calculate the angle to the cell center from the agent's position
            angle_to_cell = math.degrees(math.atan2(dy, dx))
            angle_to_cell = normalize_angle(angle_to_cell)

            # Normalize the agent's angle to 0-360 range
            agent_angle = normalize_angle(agent_angle)
            min_angle = normalize_angle(agent_angle - FOV_ANGLE / 2)
            max_angle = normalize_angle(agent_angle + FOV_ANGLE / 2)

            # Check if the angle to the cell is within the FOV range
            if min_angle < max_angle:
                if min_angle <= angle_to_cell <= max_angle:
                    visible_cells.append((cell_x, cell_y))
            else:
                # FOV spans 0 degrees, so check both segments
                if angle_to_cell >= min_angle or angle_to_cell <= max_angle:
                    visible_cells.append((cell_x, cell_y))
                    if visited_map[cell_x, cell_y] == 0:
                        count_unseen_cells += 1
                    else:
                        count_seen_cells += 1
                    visited_map[cell_x, cell_y] += 1

    print(" seen cells: " , count_seen_cells)
    print("unseen cells: " , count_unseen_cells)

    return visible_cells


# Example usage
agent_x, agent_y = 400, 400  # Example agent position on a 1000x1000 board
agent_angle = 0  # Example agent facing right (0 degrees)

visible_cells = get_cells_in_fov(agent_x, agent_y, agent_angle)

print("Cells within FOV:", visible_cells)


def is_near_border(x, y, board_size):
    border_distance = 75
    borders = [
        (0, y, 90),
        (board_size, y, -90),
        (x, 0, 180),
        (x, board_size, 0)
    ]

    for border_x, border_y, border_angle in borders:
        if self.is_point_in_fov((border_x, border_y), border_distance):
            relative_angle = self.get_relative_angle((border_x, border_y))
            if abs(relative_angle - border_angle) <= self.squint_angle:
                return True
    return False