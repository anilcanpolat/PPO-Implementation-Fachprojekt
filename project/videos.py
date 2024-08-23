import os
import argparse
import gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from random_agent import RandomAgent
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from statistics import mean


def visualize_heatmap(input_image, cam):
    if cam is None:
        if isinstance(input_image, np.ndarray):
            if input_image.ndim == 1:
                input_image = np.expand_dims(np.expand_dims(input_image, axis=0), axis=0)
            elif input_image.ndim == 2:
                input_image = np.expand_dims(input_image, axis=0)
            input_image = input_image / 255.0  # Normalize if the input range is 0-255
            return input_image

        input_image = torch.tensor(input_image)  # Convert to tensor if it's not already
        input_image = input_image.squeeze()

        if input_image.dim() == 1:  # If it's a flat array
            input_image = input_image.unsqueeze(0).unsqueeze(0)
        elif input_image.dim() == 2:  # If it's 2D, add a channel dimension
            input_image = input_image.unsqueeze(0)
        
        input_image = input_image.permute(1, 2, 0).cpu().numpy()
        input_image = input_image / 255.0  # Normalize if the input range is 0-255
        return input_image

    input_image = torch.tensor(input_image)  # Convert to tensor if it's not already
    input_image = input_image.squeeze().permute(1, 2, 0).cpu().numpy()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlayed_image = heatmap + np.float32(input_image)
    overlayed_image = overlayed_image / np.max(overlayed_image)
    return overlayed_image


def create_video(env_name, agent_class, video_path, wandb_log, episodes, live, hyperparameters):
    """Creates and runs an environment while recording every step. Plots data of the run next to the video."""
    name_prefix = "cartpole"
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_path, name_prefix=name_prefix)
    env = RecordEpisodeStatistics(env)

    if agent_class == RandomAgent:
        agent = agent_class(env.action_space)
    else:  # PPOAgent
        agent = agent_class(env, hyperparameters)
        if hyperparameters.model_path != "":
            agent.load(hyperparameters.model_path)

    if wandb_log.lower() == "true":
        wandb.init()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("Agent Playing")
    ax2.set_title("Value Function")

    max_values = []
    for episode in range(episodes):
        # Initialize the action value function plot
        action_value_data = []
        batch_observation = []
        batch_actions = []

        state, info = env.reset()
        done = False
        truncated = False
        steps = 0
        batch_observation.append(state)
        while not (done or truncated) and steps < hyperparameters.max_timesteps_per_episode:
            if live:
                env.render()
            action, _ = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)

            batch_observation.append(state)
            batch_actions.append(action)

            if hasattr(agent, 'compute_value'):
                values, _ = agent.compute_value(torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32))
                max_values.append(max_value)
                avg_value = mean(max_values)
                action_value_data.append((max_value, avg_value))
                if wandb_log.lower() == "true":
                    wandb.log({"max_value": max_value, "average_value": avg_value})
            else:
                action_value_data.append((None, None))

            # Visualize Grad-CAM
            cam = agent.visualize_gradcam(state, action)
            overlayed_image = visualize_heatmap(state, cam)

            # Ensure overlayed_image is a valid shape and normalized
            if overlayed_image.ndim == 1:
                overlayed_image = np.expand_dims(np.expand_dims(overlayed_image, axis=0), axis=0)
            elif overlayed_image.ndim == 2:
                overlayed_image = np.expand_dims(overlayed_image, axis=0)
            overlayed_image = overlayed_image.clip(0, 1)  # Ensure values are within valid RGB range

            # Update plots
            ax1.clear()
            ax1.imshow(overlayed_image)
            ax2.clear()
            ax2.set_title("Action Value Function")
            if action_value_data:
                ax2.plot([max_value], label='Max Value')
                ax2.plot([avg_value], label='Average Value')
            ax2.legend()

            plt.pause(0.01)
            steps += 1

        # Ensure video file exists
        video_file_path = os.path.join(video_path, f"{name_prefix}-episode-{episode}.mp4")
        if os.path.exists(video_file_path):
            if wandb_log.lower() == "true":
                wandb.log({"video": wandb.Video(video_file_path)})
        else:
            print(f"Warning: Video file {video_file_path} was not found.")

    env.close()
    plt.show()

"""For being run over the command line"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create videos of an agent in a Gymnasium environment.")
    # Environment parameters
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Name of the environment')
    parser.add_argument("--agent_class", type=str, default="PPOAgent", help="Name of the agent class to evaluate")
    parser.add_argument("--video_path", type=str, default="Videos", help="Path to save the video file")
    parser.add_argument("--wandb_log", type=str, default="true", help="Boolean to determine whether the data is logged to wandb")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--live", action="store_true", help="Play live in a window instead of saving to a file")
    parser.add_argument('--max_steps', type=int, default=1e6, help='Maximum number of steps for training')
    parser.add_argument('--time_limit', type=int, default=3600, help='Time limit in seconds for training')
    parser.add_argument('--max_timesteps_per_episode', type=int, default=1600, help='Maximum number of timesteps per Training episode')
    parser.add_argument('--obs_type', type=str, default='features', help='The type of the observation space ("pixels" or "features").')
    parser.add_argument("--model_path",type=str,default="",help="Path to a trained model.")

    # PPO hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='Clip epsilon for PPO')
    parser.add_argument('--update_epochs', type=int, default=10, help='Number of epochs to update the policy')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for updates')
    args = parser.parse_args()

    # Dynamically import the agent class
    if args.agent_class == "RandomAgent":
        agent_module = __import__('random_agent')
        agent_class = getattr(agent_module, args.agent_class)
    else:
        agent_module = __import__('agent')
        agent_class = getattr(agent_module, args.agent_class)

    if args.wandb_log.lower() == "true":
        wandb.init()

    create_video(env_name=args.env_name, agent_class=agent_class, video_path=args.video_path, wandb_log=args.wandb_log, episodes=args.episodes, live=args.live, hyperparameters=args)
