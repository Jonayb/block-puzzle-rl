from stable_baselines3 import PPO
from block_puzzle_gym_env import BlockPuzzleEnv


def main():
    # Create the environment
    env = BlockPuzzleEnv()

    # Instantiate the agent
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Save the agent
    model.save("block_puzzle_agent")

    # Test the trained agent
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
