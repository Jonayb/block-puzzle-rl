import numpy as np
import pygame

from block_puzzle_gym_env import BlockPuzzleEnv

env = BlockPuzzleEnv()
state, info = env.reset()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Define or randomly generate an action
        # Example: random action
        # action = (np.random.randint(0, env.width), np.random.randint(0, env.height), np.random.randint(0, env.num_blocks - 1))
        block_type = input("Enter block type: ")
        x_pos = input("Enter x position: ")
        y_pos = input("Enter y position: ")
        try:
            block_type = int(block_type)
            x_pos = int(x_pos)
            y_pos = int(y_pos)
        except ValueError:
            print("Invalid input. Please try again.")
            continue
        action = (x_pos, y_pos, block_type)
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {terminated}, Info: {info}")

        if terminated:
            print("No more moves possible. Resetting environment.")
            state, info = env.reset()
    env.render()
env.close()
