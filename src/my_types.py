from typing import NamedTuple, Dict, Any
from enum import IntEnum
import numpy as np
from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import zipfile
import io
import torch as th
from stable_baselines3 import PPO


class StepResult(NamedTuple):
  observation: np.ndarray
  reward: float
  terminated: bool
  truncated: bool
  info: Dict[str, Any]

class Action(IntEnum):
  DOWN  = 0
  LEFT  = 1
  RIGHT = 2
  UP    = 3 
  FIRE  = 4
  POLARITY = 5
  FRAGMENT = 6
  ESCAPE = 7
  PAUSE = 8
  ENTER = 9

class Keys(IntEnum):
    DOWN = 1 << 0
    LEFT = 1 << 1
    RIGHT = 1 << 2
    UP = 1 << 3
    FIRE = 1 << 4
    POLARITY = 1 << 5
    FRAGMENT = 1 << 6
    ESCAPE = 1 << 7
    PAUSE = 1 << 8
    ENTER = 1 << 9

class State(IntEnum):
  FRAME      = 0
  SCORE      = 1
  LIVES      = 2
  POWER      = 3
  POLARITY   = 4
  PLAYER_X   = 5
  PLAYER_Y   = 6
  CHAIN      = 7
  CHAIN_COLOR= 8

  RESERVED_START = 9
  RESERVED_END   = 16

  REWARD = 17

  ENEMIES    = 18 
  BULLETS    = 179

# Defini mal los primeros modelos
class NetworkVars:
    NETWORK_INPUT_VERSION = 2
    DATOS_ENTRANTES = 778
    NORMALIZE_OBSERVATION = True
    NEURONAS_ENTRANTES = 769 if NETWORK_INPUT_VERSION != 1 else 768

    @classmethod
    def set_network_input_version(cls, version: int):
        cls.NETWORK_INPUT_VERSION = version
        cls.NEURONAS_ENTRANTES = 768 if version == 1 else 769

class CustomFlatExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomFlatExtractor, self).__init__(observation_space, features_dim)
        # Assuming observation is a flat vector of 768
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)
    
    #def load_sb3_weights_manually(model_path, env):
        #local_model = PPO("MlpPolicy", env, policy_kwargs={
            #"features_extractor_class": CustomFlatExtractor,
            #"features_extractor_kwargs": {"features_dim": 256}
        #}, verbose=1)


        #local_model.policy.load_state_dict(state_dict, strict=False)
        #print("Successfully loaded weights manually!")
        #return local_model


class CustomFlatExtractor2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(NetworkVars.neuronas_entrantes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
            nn.ReLU()
        )

    def forward(self, x):
          x = self.flatten(x)
          logits = self.linear_relu_stack(x)
          return logits

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(NetworkVars.NEURONAS_ENTRANTES, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
    )
  def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class PPOActorNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, features_dim: int = 256):
        super().__init__()

        # === CustomFlatExtractor ===
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

        # === Actor MLP (from MlpExtractor.policy_net) ===
        self.actor_mlp = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        # === Action head ===
        self.action_head = nn.Linear(64, action_dim)

    def forward(self, obs: th.Tensor):
        """
        Returns action logits (discrete) or action means (continuous).
        """
        features = self.feature_extractor(obs)
        latent_pi = self.actor_mlp(features)
        return self.action_head(latent_pi)