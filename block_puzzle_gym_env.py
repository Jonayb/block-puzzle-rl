from typing import Tuple, Dict

import pygame
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from blocks import blocks


class BlockPuzzleEnv(gym.Env):
    """
    Custom Environment that follows gym interface for a 9x9 block puzzle game.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BlockPuzzleEnv, self).__init__()
        # Define action and observation space
        self.blocks = blocks
        self.block_ids = list(self.blocks.keys())
        self.num_blocks = len(self.blocks)
        self.action_space = gym.spaces.MultiDiscrete([9, 9, 3])
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(9, 9), dtype=np.int32),  # The 9x9 grid
            gym.spaces.MultiBinary(self.num_blocks)  # The 3-element vector for blocks
        ))
        self.state = None
        self.reset()

        pygame.init()
        self.screen_size = 600
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        self.cell_size = self.screen_size / 15

    def step(self, action):
        block_type, pos_x, pos_y = action
        reward = 0
        done = False
        info = {}

        # Check if the block type is available
        if self.state[1][block_type] == 0:
            reward -= 1  # Small punishment for choosing an unavailable block
            return self.state, reward, done, info

        # Get block coordinates
        block_coords = self.blocks[block_type]["coords"]

        # Check for feasibility
        if not self.is_placement_feasible(block_coords, pos_x, pos_y):
            reward -= 1  # Small punishment for infeasible placement
            return self.state, reward, done, info

        # Place the block on the grid
        for coord in block_coords:
            x, y = pos_x + coord[0], pos_y + coord[1]
            self.state[0][y][x] = 1
        reward += len(block_coords)  # Reward proportional to the block size

        # Update block availability
        self.state[1][block_type] = 0

        # Resample blocks if all are used
        if not self.state[1].any():
            self.state[1] = np.random.choice([0, 1], size=self.num_blocks, p=[0.8, 0.2])
            self.state[1][:3] = 1  # Ensure at least 3 blocks are available

        # Check and clear full rows, columns, and 3x3 areas
        reward += self.check_and_clear_full_areas()

        # Check if no moves are possible
        if not self.any_feasible_moves():
            done = True

        return self.state, reward, done, info

    def check_and_clear_full_areas(self):
        additional_reward = 0

        # Check rows and columns
        for i in range(9):
            if all(self.state[0][i, :]):  # Check if row i is full
                self.state[0][i, :] = 0
                additional_reward += 27
            if all(self.state[0][:, i]):  # Check if column i is full
                self.state[0][:, i] = 0
                additional_reward += 27

        # Check 3x3 areas
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                area = self.state[0][i:i+3, j:j+3]
                if np.all(area == 1):
                    self.state[0][i:i+3, j:j+3] = 0
                    additional_reward += 27

        return additional_reward

    def is_placement_feasible(self, block_coords, pos_x, pos_y):
        for coord in block_coords:
            x, y = pos_x + coord[0], pos_y + coord[1]
            if x >= 9 or y >= 9 or self.state[0][y][x] == 1:
                return False
        return True

    def any_feasible_moves(self):
        for block_type, available in enumerate(self.state[1]):
            if available:
                block_coords = self.blocks[block_type]["coords"]
                for x in range(9):
                    for y in range(9):
                        if self.is_placement_feasible(block_coords, x, y):
                            return True
        return False

    def reset(self, seed=None, options=None):
        if seed:
            random.seed(seed)
        # Reset the state of the environment to an initial state
        state = np.zeros((9, 9), dtype=np.int32)
        # Randomly sample three blocks
        sampled_blocks = np.random.choice(self.block_ids, 3)
        # Update the block availability vector in the state
        block_availability = np.zeros(self.num_blocks, dtype=np.int32)
        for block in sampled_blocks:
            block_availability[block] = 1

        # Combine the grid state and block availability into the complete state
        self.state = (state, block_availability)
        return self.state, {}

    def render(self, mode='human'):
        grid, block_availability = self.state

        # Set background color
        self.screen.fill((255, 255, 255))  # White background

        # Draw the 9x9 grid
        for x in range(9):
            for y in range(9):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Black border for each cell

                # Fill in cells if they are occupied
                if grid[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Fill cell with black

        block_count = 0
        # Render the available blocks below the grid
        for idx, available in enumerate(block_availability):
            if available:
                block = self.blocks[idx]
                block_coords = block["coords"]
                for coord in block_coords:
                    block_rect = pygame.Rect(
                        10 + (coord[0] + block_count * 6) * self.cell_size / 2,
                        self.cell_size * 10 + coord[1] * self.cell_size / 2,
                        self.cell_size / 2, self.cell_size / 2
                    )
                    pygame.draw.rect(self.screen, (0, 0, 255), block_rect)  # Blue blocks for simplicity
                block_count += 1

        # Update the display
        pygame.display.flip()

    def close(self):
        pygame.quit()


# Test the environment
if __name__ == "__main__":
    env = BlockPuzzleEnv()
    env.reset()
    print(env.reset())
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        env.render()
    env.close()
