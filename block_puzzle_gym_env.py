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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super(BlockPuzzleEnv, self).__init__()
        # Define action and observation space
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.width = 9
        self.height = 9
        self.action_space = gym.spaces.MultiDiscrete([self.width, self.height, self.num_blocks])
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.int32),  # The 9x9 grid
            gym.spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int32),  # First 5x5 block matrix
            gym.spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int32),  # Second 5x5 block matrix
            gym.spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int32)  # Third 5x5 block matrix
        ))
        self.current_blocks = [None, None, None]
        self.grid = None
        self.step_count = 0
        self.streak = 0
        self.score = 0
        self.reset()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 500
        self.window = None
        self.clock = None
        self.cell_size = self.window_size // 15

    def step(self, action):
        pos_x, pos_y, block_choice_index = action
        reward = 0
        terminated = False
        self.step_count += 1

        # Check if the step_count limit has been reached
        if self.step_count > 100_000:
            return self._get_obs(), reward, terminated, True, self._get_info()

        # Check if the block choice is valid
        if self.current_blocks[block_choice_index] is None:
            reward -= 1
            return self._get_obs(), reward, terminated, False, self._get_info()

        # Get block coordinates
        block_id = self.current_blocks[block_choice_index]
        block_coords = self.blocks[block_id]["coords"]

        # Check for feasibility
        if not self._is_placement_feasible(block_coords, pos_x, pos_y):
            reward -= 1  # Small punishment for infeasible placement
            return self._get_obs(), reward, terminated, False, self._get_info()

        # Place the block on the grid
        for coord in block_coords:
            x, y = pos_x + coord[0], pos_y + coord[1]
            self.grid[y][x] = 1

        # Update block availability
        self.current_blocks[block_choice_index] = None

        # Resample blocks if all are used
        if not any(self.current_blocks):
            self.current_blocks = list(np.random.choice(self.num_blocks, 3))  # Select 3 unique blocks

        # Check and clear full rows, columns, and 3x3 areas
        reward += self.check_and_clear_full_areas()

        # Check if no moves are possible
        if not self._any_feasible_moves():
            terminated = True

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        # Construct the observation state with a 9x9 grid and 5x5 matrices
        matrices = [self.blocks[block_id]["matrix"] if block_id is not None else np.zeros((5, 5), dtype=np.int32)
                    for block_id in self.current_blocks]
        return self.grid, *matrices

    def check_and_clear_full_areas(self):
        reward = 0
        num_cleared = 0
        num_occupied = np.sum(self.grid)
        coords_to_clear = []

        # Check rows and columns
        for i in range(9):
            if all(self.grid[i, :]):  # Check if row i is full
                num_cleared += 1
                coords_to_clear.extend([(x, i) for x in range(9)])
            if all(self.grid[:, i]):  # Check if column i is full
                num_cleared += 1
                coords_to_clear.extend([(i, y) for y in range(9)])

        # Check 3x3 areas
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                area = self.grid[i:i+3, j:j+3]
                if np.all(area == 1):
                    num_cleared += 1
                    coords_to_clear.extend([(i + x, j + y) for x in range(3) for y in range(3)])

        if num_cleared > 0:
            self.streak += 1
            if num_cleared == 1:
                reward += self.streak * 29
            elif num_cleared == 2:
                reward += self.streak * 71
            elif num_cleared == 3:
                reward += self.streak * 131
            else:
                reward += self.streak * 211
        else:
            self.streak = 0

        # Clear the full rows, columns, and 3x3 areas
        for x, y in coords_to_clear:
            self.grid[y][x] = 0

        reward += (np.sum(self.grid) + len(coords_to_clear) - num_occupied)
        self.score += reward
        return reward

    def _is_placement_feasible(self, block_coords, pos_x, pos_y):
        for coord in block_coords:
            x, y = pos_x + coord[0], pos_y + coord[1]
            if x >= 9 or y >= 9 or self.grid[y][x] == 1:
                return False
        return True

    def _any_feasible_moves(self):
        for block_id in self.current_blocks:
            if block_id is not None:
                block_coords = self.blocks[block_id]["coords"]
                for x in range(9):
                    for y in range(9):
                        if self._is_placement_feasible(block_coords, x, y):
                            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((9, 9), dtype=np.int32)
        self.current_blocks = list(np.random.choice(self.num_blocks, 3))

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        grid, *block_matrices = self._get_obs()  # Unpack the grid and block matrices

        # Draw the 9x9 grid
        for x in range(9):
            for y in range(9):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

                # Draw the 3x3 areas
                if x % 3 == 0 and y % 3 == 0:
                    pygame.draw.rect(canvas,
                                     (0, 0, 0),
                                     pygame.Rect(x * self.cell_size,
                                                 y * self.cell_size,
                                                 self.cell_size * 3,
                                                 self.cell_size * 3),
                                     3)

                # Fill in cells if they are occupied
                if grid[y][x] == 1:
                    pygame.draw.rect(canvas, (0, 0, 0), rect)  # Fill cell with black

        # Render the available blocks below the grid
        for block_idx, block_matrix in enumerate(block_matrices):
            # Check if the block is available
            if np.any(block_matrix):
                for i in range(5):
                    for j in range(5):
                        if block_matrix[i][j] == 1:
                            block_rect = pygame.Rect(
                                10 + (j + block_idx * 6) * self.cell_size / 2,
                                self.cell_size * 10 + i * self.cell_size / 2,
                                self.cell_size / 2, self.cell_size / 2
                            )
                            pygame.draw.rect(canvas, (0, 0, 255), block_rect)  # Blue blocks for simplicity

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _get_info(self):
        return {"step_count": self.step_count,
                "streak": self.streak,
                "blocks": self.current_blocks,
                "score": self.score}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
