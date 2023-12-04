import numpy as np
import pygame
from stable_baselines3 import PPO

from block_puzzle_gym_env import BlockPuzzleEnv

env = BlockPuzzleEnv(render_mode="human")
state, info = env.reset()
running = True

# Instantiate the agent
model = PPO.load("block_puzzle_agent", env=env)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    action, _ = model.predict(state, deterministic=False)
    state, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {terminated}, Info: {info}")

    if terminated:
        print("No more moves possible. Resetting environment.")
        state, info = env.reset()

env.close()
