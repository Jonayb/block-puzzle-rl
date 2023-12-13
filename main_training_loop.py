from block_puzzle_gym_env import BlockPuzzleEnv

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import MaskablePPO


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):  # Increased features_dim
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Define a deeper CNN for the 9x9 grid
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Flatten()
        )

        # Calculate the flattened size
        dummy_input = th.zeros((1, 1) + observation_space.shape, dtype=th.float32)
        n_flatten = self.cnn(dummy_input).shape[1]

        # Fully connected layer
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations.unsqueeze(1)))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=1024),
    net_arch=[1024, 1024],
)


def main():
    # Create the environment
    env = BlockPuzzleEnv()

    # Set the TensorBoard log directory
    tensorboard_log = "./block_puzzle_tensorboard/"

    # Instantiate the agent
    model = MaskablePPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log)
    print(model.policy)

    # Train the agent
    model.learn(total_timesteps=1_000_000)

    # Save the agent
    model.save("block_puzzle_agent")


if __name__ == '__main__':
    main()
