from block_puzzle_gym_env import BlockPuzzleEnv

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Define the CNN for the 9x9 grid
        self.cnn_grid = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Define the CNN for the 5x5 blocks
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of the flattened features from each part
        with torch.no_grad():
            n_flatten_grid = self.cnn_grid(torch.as_tensor(observation_space['grid'].sample()[None, None, :, :]).float()).shape[1]
            n_flatten_block = self.cnn_block(torch.as_tensor(observation_space['first_block'].sample()[None, None, :, :]).float()).shape[1]

        # Fully connected layer to combine features
        self.fc = nn.Linear(n_flatten_grid + 3 * n_flatten_block, features_dim)

    def forward(self, observations):
        grid = observations['grid']
        block1 = observations['first_block']
        block2 = observations['second_block']
        block3 = observations['third_block']

        # Reshape the inputs for CNN compatibility
        grid = grid.unsqueeze(1)  # Add channel dimension
        block1 = block1.unsqueeze(1)
        block2 = block2.unsqueeze(1)
        block3 = block3.unsqueeze(1)

        # Pass through CNNs
        grid_features = self.cnn_grid(grid)
        block1_features = self.cnn_block(block1)
        block2_features = self.cnn_block(block2)
        block3_features = self.cnn_block(block3)

        # Concatenate features and pass through fully connected layer
        combined_features = torch.cat([grid_features, block1_features, block2_features, block3_features], dim=1)
        return self.fc(combined_features)


class CustomMultiInputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMultiInputPolicy, self).__init__(*args, **kwargs,
                                                     features_extractor_class=CustomCNNFeatureExtractor,
                                                     features_extractor_kwargs={'features_dim': 128}
                                                     )


def main():
    # Create the environment
    env = BlockPuzzleEnv()

    # Set the TensorBoard log directory
    tensorboard_log = "./block_puzzle_tensorboard/"

    # Instantiate the agent
    model = PPO(CustomMultiInputPolicy, env, verbose=1, tensorboard_log=tensorboard_log)

    # Train the agent
    model.learn(total_timesteps=1_000_000)

    # Save the agent
    model.save("block_puzzle_agent")


if __name__ == '__main__':
    main()
