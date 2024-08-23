import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import gym
from networks import ActorCritic
from torch.optim import Adam
from torch.distributions import Categorical
from env_wrappers import PreprocessingWrapper, FrameStackingWrapper
import wandb
import os

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = dict([*self.model.named_modules()])[target_layer_name]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

        # Print all layers to help identify correct target layer
        print("Available layers:")
        for name, layer in self.model.named_modules():
            print(name)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, action):
        # Forward pass
        model_output = self.model(input_image)
        policy_output = model_output[0]  # Assuming model_output is a tuple (policy, value)
        
        # Ensure policy_output is at least 2D
        if policy_output.dim() == 1:
            policy_output = policy_output.unsqueeze(0)
        
        one_hot_output = torch.zeros_like(policy_output)
        one_hot_output[0, action] = 1

        # Backward pass
        self.model.zero_grad()
        policy_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get the gradients and feature maps
        gradients = self.gradients.cpu().data.numpy()
        target = self.activations.cpu().data.numpy()
        
        # Debug prints to check dimensions
        print(f"gradients shape: {gradients.shape}")
        print(f"target shape: {target.shape}")

        if gradients.ndim == 2:
            weights = np.mean(gradients, axis=1)  # Adjust for non-spatial layers
        else:
            weights = np.mean(gradients, axis=(1, 2))

        if target.ndim == 2:
            cam = weights  # Adjust for non-spatial layers
        else:
            cam = np.zeros(target.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (input_image.size(2), input_image.size(3)))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

        return cam

class PPOAgent:
    def __init__(self, env, hyperparameters):
        self.env = env
        self.hyperparameters = hyperparameters
        if hyperparameters.obs_type == "pixels":
            self.env = PreprocessingWrapper(self.env)
            self.env = FrameStackingWrapper(self.env)
            self.observation_dim = self.env.observation_space.shape
        else:
            self.observation_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.action_dim = sum(env.action_space.nvec)
        elif isinstance(env.action_space, gym.spaces.MultiBinary):
            self.action_dim = env.action_space.n
        else:
            raise ValueError("Unsupported action space type")

        self.actorcritic = ActorCritic(self.observation_dim, self.action_dim, self.hyperparameters.obs_type)
        self.actorcritic_optim = Adam(self.actorcritic.parameters(), lr=self.hyperparameters.learning_rate)

        conv_layers = self.find_conv_layers(self.actorcritic)
        if conv_layers:
            target_layer = conv_layers[-1]
            self.grad_cam = GradCAM(self.actorcritic, target_layer)
        else:
            self.grad_cam = None  # or handle the case where no conv layer is found

    def find_conv_layers(self, model):
        conv_layers = []
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                conv_layers.append(name)
        return conv_layers

    def learn(self, total_timesteps):
        wandb.init()
        current_timestep = 0
        current_iteration = 0
        checkpoint_interval = 25000
        policy_loss_array = []
        value_loss_array = []
        
        while current_timestep < total_timesteps:
            obs, log_prob, rewards, dones, values, actions = self.rollout()
            advantages, value_targets = self.compute_gae_value_target(dones, rewards, values)
    
            obs = torch.FloatTensor(np.array(obs))
            actions = torch.LongTensor(actions)
            log_probs = torch.FloatTensor(log_prob)
            value_targets = torch.FloatFloat(value_targets)
            advantages = torch.FloatTensor(advantages)
    
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
           

            for _ in range(self.hyperparameters.update_epochs):
                for start in range(0, len(obs), self.hyperparameters.batch_size):
                    end = start + self.hyperparameters.batch_size
                    batch_obs = obs[start:end]
                    batch_actions = actions[start:end]
                    batch_log_probs = log_probs[start:end]
                    batch_value_targets = value_targets[start:end]
                    batch_advantages = advantages[start:end]
                    
                    if self.hyperparameters.obs_type == "pixels":
                        for batch in batch_obs:
                                policy_logits, values = self.actorcritic(batch)
                                all_policy_logits.append(policy_logits)
                                all_values.append(values)
                        policy_logits = torch.stack(all_policy_logits)
                        values = torch.stack(all_values)
                    else:
                        policy_logits, values = self.actorcritic(batch_obs)
                    policy_dist = Categorical(logits=policy_logits)
                    new_log_probs = policy_dist.log_prob(batch_actions)
                    entropy = policy_dist.entropy().mean()

                    ratio = (new_log_probs - batch_log_probs).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.hyperparameters.clip_epsilon, 1 + self.hyperparameters.clip_epsilon) * batch_advantages

                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(values.squeeze(), batch_value_targets)

                    policy_loss_array.append(policy_loss.item())
                    value_loss_array.append(value_loss.item())

                    loss = policy_loss + 0.1 * value_loss

                    self.actorcritic_optim.zero_grad()
                    loss.backward()
                    self.actorcritic_optim.step()

                    # define our custom x axis metric
                    wandb.define_metric("current_timestep")
                    # define which metrics will be plotted against it
                    wandb.define_metric("policy_loss", step_metric="current_timestep")
                    wandb.define_metric("value_loss", step_metric="current_timestep")
                    wandb.log({'policy_loss': np.mean(policy_loss_array), 'current_timestep': current_timestep})
                    wandb.log({'value_loss': np.mean(value_loss_array), 'current_timestep': current_timestep})

                    if current_timestep >= checkpoint_interval:
                        checkpoint_dir = "./checkpoints"
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{current_timestep}.pth")
                        self.save(checkpoint_path)
                        checkpoint_interval += 25000
                        print(f"Checkpoint saved at step {current_timestep}.")

            current_timestep += len(obs)
            current_iteration += 1
        return total_timesteps

    def rollout(self):
        obs, log_prob, rewards, dones, values, actions = [], [], [], [], [], []
        if self.hyperparameters.obs_type == "pixels":
            observation = self.env.reset()
        else:
            observation, _ = self.env.reset()

        for step in range(self.hyperparameters.max_timesteps_per_episode):
            policy_logits, value = self.actorcritic(observation)
            action, log_proba = self.select_action(observation)
            next_obs, reward, done, truncated, _ = self.env.step(action)

            obs.append(observation)
            observation = next_obs
            dones.append(done)
            log_prob.append(log_proba.detach())
            values.append(value.detach())
            actions.append(action)
            rewards.append(reward)

            if done or truncated:
                if self.hyperparameters.obs_type == "pixels":
                    observation = self.env.reset()
                else:
                    observation, _ = self.env.reset()

        return obs, log_prob, rewards, dones, values, actions

    def select_action(self, observation):
        mean, _ = self.actorcritic(observation)
        mean = torch.softmax(mean, dim=-1)
        dist = Categorical(mean)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().item(), log_prob.detach()

    def compute_gae_value_target(self, dones, rewards, values):
        last_value = values[-1]
        last_adv = 0
        advantages = []
        value_targets = []
        gae = 0

        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.hyperparameters.gamma * values[t + 1] * mask - values[t]
            gae = last_adv * self.hyperparameters.gamma * self.hyperparameters.lambda_gae + delta
            advantages.append(gae)
            last_adv = gae
            value_target = rewards[t] + last_value * self.hyperparameters.gamma * mask
            value_targets.append(value_target)
            last_value = value_target
        return list(reversed(advantages)), list(reversed(value_targets))
    
    def compute_value(self, batch_observation, batch_actions):
        _, values = self.actorcritic(batch_observation)
        values = values.squeeze()
        mean, _ = self.actorcritic(batch_observation)
        mean = torch.softmax(mean, dim=-1)
        dist = Categorical(mean)
        log_probs = dist.log_prob(batch_actions)
        return values, log_probs

    def save(self, path):
        torch.save(self.actorcritic.state_dict(), path)

    def load(self, path):
        self.actorcritic.load_state_dict(torch.load(path))

    def visualize_gradcam(self, input_image, action):
        if self.grad_cam:
            cam = self.grad_cam.generate_cam(input_image, action)
            return cam
        else:
            print("Grad-CAM not available, as no convolutional layer was found.")
            return None

