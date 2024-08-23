import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    """Network for the PPO implementation"""
    def __init__(self, input_dim, output_dim, obs_type):
        super(ActorCritic, self).__init__()
        self.obs_type = obs_type

        if self.obs_type == "features":
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.actor_fc = nn.Linear(128, output_dim)
            self.critic_fc = nn.Linear(128, 1)
        else:
            height, width, frames = input_dim
            
            self.conv1 = nn.Conv2d(in_channels=frames, out_channels=32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            # Calculate the output size after the conv layers
            self.fc_input_dim = self._get_conv_output(height, width)    
            # Fully connected layer
            self.fc = nn.Linear(self.fc_input_dim, 512)
            
            self.actor_fc = nn.Linear(512, output_dim)
            self.critic_fc = nn.Linear(512, 1)

    def _get_conv_output(self, height, width):
       def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        out_height = conv2d_size_out(height, 8, 4)  # After conv1
        out_height = conv2d_size_out(out_height, 4, 2)  # After conv2
        out_height = conv2d_size_out(out_height, 3, 1)  # After conv3

        out_width = conv2d_size_out(width, 8, 4)  # After conv1
        out_width = conv2d_size_out(out_width, 4, 2)  # After conv2
        out_width = conv2d_size_out(out_width, 3, 1)  # After conv3

        return out_height * out_width * 64  # 64 is the number of channels in the third conv layer

    def forward(self, x):
        if self.obs_type == "features":
            x = torch.tensor(np.array(x), dtype=torch.float) 
            
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            
            policy = self.actor_fc(x)
            value = self.critic_fc(x)
        else:
           x = torch.tensor(observation, dtype=torch.float)
            if x.dim() == 3:  # Single observation case
                x = x.unsqueeze(0)  # Add batch dimension
            if x.dim() == 4:  # Expecting shape (batch_size, height, width, frames)
                x = x.permute(0, 3, 1, 2)  # Change shape to (batch_size, frames, height, width)
            elif x.dim() == 5:  # Expecting shape (batch_size, height, width, channels, frames)
                x = x.permute(0, 4, 3, 1, 2)  # Change shape to (batch_size, frames, channels, height, width)
            else:
                raise ValueError(f"Unexpected input dimensions: {x.shape}")
            
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc(x))
            
            policy = self.actor_fc(x)
            value = self.critic_fc(x)
        return policy, value
