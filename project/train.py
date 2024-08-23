import argparse
import gym
import torch
import os
from agent import PPOAgent
import time

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Training Script')
    
    # Environment parameters
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Name of the environment')
    parser.add_argument('--max_steps', type=int, default=1e6, help='Maximum number of steps for training')
    parser.add_argument('--time_limit', type=int, default=3600, help='Time limit in seconds for training')
    parser.add_argument('--max_timesteps_per_episode', type=int, default=1600, help='Maximum number of timesteps per Training episode')
    parser.add_argument('--obs_type', type=str, default='features', help='The type of the observation space ("pixels" or "features").')

    # PPO hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='Clip epsilon for PPO')
    parser.add_argument('--update_epochs', type=int, default=10, help='Number of epochs to update the policy')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for updates')
    parser.add_argument('--lambda_gae', type=float, default=0.95, help='Lambda for GAE')
    

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    env = gym.make(args.env_name)
    agent = PPOAgent(env, args)

    # Load existing checkpoint if available
    checkpoint_path = "./checkpoints/checkpoint_last.pth"
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)

    start_time = time.time()
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    total_steps = 0

    while (time.time() - start_time) < args.time_limit:
        steps = agent.learn(args.max_steps)
        print("steps:", total_steps)
        if steps is None:
            print("Error: The agent's learn method did not return the number of steps taken.")
            break
        total_steps += steps

        # Save checkpoint at intervals                                                        Saving checkpoints in agent
        #if total_steps >= checkpoint_interval:
        #    agent.save(os.path.join(checkpoint_dir, f"checkpoint_{total_steps}.pth"))
        #    checkpoint_interval += 5000

        # Check if the maximum number of steps has been reached
        if total_steps >= args.max_steps:
            break
    # Save final checkpoint
    agent.save(os.path.join(checkpoint_dir, "checkpoint_last.pth"))
    env.close()

if __name__ == '__main__':
    main()
