from time import sleep

from stable_baselines3 import PPO
from block_puzzle_gym_env import BlockPuzzleEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Assuming the grid is 9x9 and the block vector is of length num_blocks
        num_blocks = observation_space.shape[0] - 81

        # CNN for the grid
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1 input channel, 16 output channels
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output for the fully connected layers
        )

        # Fully connected network for the block vector
        self.fc = nn.Sequential(
            nn.Linear(num_blocks, 16),
            nn.ReLU(),
        )

        # Combined output layer
        self.combined_fc = nn.Linear(32, features_dim)  # 16 from CNN and 16 from FC

    def forward(self, observations):
        grid = observations[:, :81].view(-1, 1, 9, 9)  # Reshape grid to match CNN input
        blocks = observations[:, 81:]

        grid_features = self.cnn(grid)
        block_features = self.fc(blocks)

        combined = torch.cat([grid_features, block_features], dim=1)
        return self.combined_fc(combined)


# Define a custom Actor-Critic policy
class CustomActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomCNNFeatureExtractor(self.observation_space, features_dim=64)


def main():
    # Create the environment
    env = BlockPuzzleEnv()

    # Instantiate the agent
    model = PPO(CustomActorCriticPolicy, env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=1_000_000)

    # Save the agent
    model.save("block_puzzle_agent")

    # Test the trained agent
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {terminated}, Info: {info}")
        sleep(0.1)
        env.render()


if __name__ == '__main__':
    main()
