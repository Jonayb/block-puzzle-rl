import numpy as np
import pygame

from block_puzzle_gym_env import BlockPuzzleEnv

env = BlockPuzzleEnv(render_mode="human")
state, info = env.reset()
running = True
random = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Define or randomly generate an action
        # Example: random action
        if random or input("Random action? (y/n): ") == "y":
            random = True
        if random:
            action = (np.random.randint(0, env.width), np.random.randint(0, env.height), np.random.randint(0, 3))
        else:
            block_choice = input("Enter block choice: ")
            x_pos = input("Enter x position: ")
            y_pos = input("Enter y position: ")
            try:
                block_choice = int(block_choice)
                x_pos = int(x_pos)
                y_pos = int(y_pos)
            except ValueError:
                print("Invalid input. Please try again.")
                continue
            action = (x_pos, y_pos, block_choice)
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {terminated}, Info: {info}")

        if terminated:
            print("No more moves possible. Resetting environment.")
            state, info = env.reset()
env.close()
