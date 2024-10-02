import numpy as np
import cv2  # Import OpenCV
from snake_environment import SnakesWorldEnv

if __name__ == "__main__":
    env = SnakesWorldEnv()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Specify codec
    video_writer = cv2.VideoWriter('snake_movement.avi', fourcc, 20.0,
                                   (1000, 1000))  # Video dimensions should match your environment

    obs = env.reset()

    for _ in range(100):  # Run for 100 steps
        action = np.random.uniform(-30, 30)  # Random action within the specified range
        obs, reward, done, success, _ = env.step(action)

        # Print debug output
        print(f"Reward: {reward}, Done: {done}")
        print(f"Snake Position: {env.snake_head_position}, Goal Position: {env.goal_position}")

        # Create an image frame for the current state
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)  # Create a black frame
        head_position = env.snake_head_position.astype(int)
        cv2.circle(frame, (head_position[0], head_position[1]), 10, (0, 255, 0), -1)  # Green head

        # Calculate the endpoint of the snake based on its length and angle
        radians = np.deg2rad(env.snake_head_angle)
        end_x = int(head_position[0] + 100 * np.cos(radians))
        end_y = int(head_position[1] + 100 * np.sin(radians))

        # Draw the snake's length as a line
        cv2.line(frame, (head_position[0], head_position[1]), (end_x, end_y), (0, 255, 0), 2)

        # Draw the goal position
        goal_position = env.goal_position.astype(int)
        cv2.circle(frame, (goal_position[0], goal_position[1]), 10, (0, 0, 255), -1)  # Red goal
        # Write the frame to the video
        video_writer.write(frame)

        if done:
            break

    video_writer.release()  # Release the VideoWriter
    cv2.destroyAllWindows()  # Close any OpenCV windows

    print("Video saved as snake_movement.avi")
