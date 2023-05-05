import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
from functools import partial
import os
import time 
import ppo.utils as utils

class PPO(nn.Module):
    def __init__(self, observation_space, nOutputNum=12, features_dim=512):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():            
            data = torch.randn(observation_space)
            n_flatten = self.cnn(data[None]).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
        self.action_net = nn.Linear(features_dim,nOutputNum)
        self.value_net  = nn.Linear(features_dim,1)
        
        #self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        module_gains = {
            self.cnn: np.sqrt(2),
            self.linear: np.sqrt(2),
            self.action_net: 0.01,
            self.value_net: 1,
        }
        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

    @staticmethod
    def init_weights(module, gain=1):
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)                
    @property
    def device(self):
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.
        :return:"""
        for param in self.parameters():
            return param.device
        return utils.get_device("cpu")

    def forward(self, obs):
        # Preprocess the observation if needed
        obs = utils.preprocess_obs(obs)
        features = self.cnn(obs)
        features = self.linear(features)
        # Evaluate the values for the given observations
        values = self.value_net(features)
        action_logits = self.action_net(features)
        #print(action_logits)
        distribution = Bernoulli(logits=action_logits)
        actions = distribution.sample()        
        log_prob = distribution.log_prob(actions).sum(dim=1)
        return actions, values, log_prob
        
    def predict(self, obs):
        # Preprocess the observation if needed
        obs = utils.preprocess_obs(obs)
        features = self.cnn(obs)
        features = self.linear(features)
        action_logits = self.action_net(features)
        #print(action_logits)
        distribution = Bernoulli(logits=action_logits)
        actions = distribution.sample()        
        return actions
        
    def predict_values(self, obs):    
        obs = utils.preprocess_obs(obs)
        features = self.cnn(obs)
        features = self.linear(features)
        return self.value_net(features)    
        
    def evaluate_actions(self, obs, actions):
        # Preprocess the observation if needed
        obs = utils.preprocess_obs(obs)
        features = self.cnn(obs)
        features = self.linear(features)
        values = self.value_net(features)
        action_logits = self.action_net(features)
        distribution = Bernoulli(logits=action_logits)
        log_prob = distribution.log_prob(actions).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)        
        return values, log_prob, entropy
     
    def configure_optimizers(self, lr, eps=1e-5): 
        return torch.optim.Adam(self.parameters(), lr=lr, eps=eps)

if __name__ == '__main__':
    model = PPO([3,100,128])
    print(model.device)
    print(model)
    data = torch.empty([1,100,128,3]).random_(to=256)
    print(data, data.shape)
    actions, values, log_prob = model(data)
    print(actions, values, log_prob)    
    value = model.predict_values(data)
    print(value)
    ac = torch.empty(12).random_(to=2)
    print(ac)
    values, log_prob, entropy = model.evaluate_actions(data, ac)
    print(values, log_prob, entropy)
    
    ac = actions
    values, log_prob, entropy = model.evaluate_actions(data, ac)
    print(values, log_prob, entropy)
    
    