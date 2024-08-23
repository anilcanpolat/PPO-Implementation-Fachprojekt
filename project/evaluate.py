import argparse
import gym
import random_agent
import wandb
from random_agent import RandomAgent
from agent import PPOAgent
from env_wrappers import PreprocessingWrapper, FrameStackingWrapper, RewardClippingWrapper

def evaluate_agent(env_name, agent_class, wandb_log, hyperparameters, num_episodes=10, ):
    """Creates an environment, runs it and calculates the mean sum of rewards over all episodes of the selected agent."""

    env = gym.make(env_name)
    if hyperparameters.obs_type == "pixels":
        env = RewardClippingWrapper(env)
        env = PreprocessingWrapper(env)
        env = FrameStackingWrapper(env)

    #Checking how the agent should be initialised
    if agent_class == RandomAgent:
        agent = agent_class(env.action_space)
    else: #PPOAgent
        agent = agent_class(env,hyperparameters)
        if hyperparameters.model_path != "":
            agent.load(hyperparameters.model_path)            # add here the pth u have trained

    total_rewards = 0

    #Running the environment
    for episode in range(num_episodes):
        if hyperparameters.obs_type == "pixels":
            state = env.reset()
        else:
            state,_ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            if agent_class == RandomAgent:
                action = agent.select_action(state)
            else: 
                action,_ = agent.select_action(state)
            if hyperparameters.obs_type == "pixels":
                state, reward, done, truncated = env.step(action)
            else 
                state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        total_rewards += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    env.close()
    #Calculating mean sum of rewards
    avg_reward = total_rewards / num_episodes

    if wandb_log == "True" or wandb_log == "true":
        wandb.log({"mean_sum_of_rewards":avg_reward})
        
    print(f'Average reward over {num_episodes} episodes: {avg_reward}')
   

"""For being run over the command line"""
if __name__ == "__main__":
    #Initialising command line arguments
    parser = argparse.ArgumentParser(description="Evaluate an agent in a Gymnasium environment.")
    # Environment parameters
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Name of the environment')
    parser.add_argument('--max_steps', type=int, default=1e6, help='Maximum number of steps for training')
    parser.add_argument('--time_limit', type=int, default=3600, help='Time limit in seconds for training')
    parser.add_argument('--max_timesteps_per_episode', type=int, default=1600, help='Maximum number of timesteps per Training episode')
    parser.add_argument('--obs_type',type=str,default='features',help='The type of the observation space ("pixels" or "features").')
    parser.add_argument("--agent_class", type=str,default="PPOAgent", help="Name of the agent class to evaluate")
    parser.add_argument("--wandb_log",type=str, default="true", help="Boolean to determine whether the data is logged to wandb")
    parser.add_argument("--model_path",type=str,default="",help="Path to a trained model.")

    # PPO hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='Clip epsilon for PPO')
    parser.add_argument('--update_epochs', type=int, default=10, help='Number of epochs to update the policy')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for updates')
    parser.add_argument('--lambda_gae', type=float, default=0.95, help='Batch size for updates')

    args = parser.parse_args()

    # Dynamically import the agent class
    if args.agent_class == "RandomAgent":
        agent_module = __import__('random_agent')
        agent_class = getattr(agent_module, args.agent_class)
    else:
        agent_module = __import__('agent')
        agent_class = getattr(agent_module,args.agent_class)

    # Determine whether data should be logged to w&b
    if args.wandb_log == "True" or args.wandb_log == "true":
        wandb.init()

    evaluate_agent(env_name=args.env_name, agent_class=agent_class, wandb_log=args.wandb_log, hyperparameters=args)
