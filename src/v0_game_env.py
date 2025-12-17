import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th

from v0_game_hooks import GameController
from utils import Utils
from my_types import StepResult, State, Action
from agents import Agents

register(
  id='MyGame-v0',
  entry_point='v0_game_env:GameEnv',
)


class GameEnv(gym.Env):
  metadata = {'render_modes': ['human'], 'render_fps': 1}

  def __init__(self, render_mode=None):
    self.reset()
    self.renderMode = render_mode
    if(self.renderMode):
      controller = GameController(headless=False)
    else:
      controller = GameController(headless=True)


    controller.start()
    self.controller = controller
    self.max_steps = 20
    self.current_step = 0

    # action space
    self.action_space = spaces.MultiBinary(len(Action))

    # observation space
    # TODO - define more precise observation space
    self.observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(768,),
    dtype=np.float32
    )
  
  def reset(self, seed=None, options=None):
    print("ENV RESET CALLED")
    super().reset(seed=seed)
    # initialize value - i think i dont have these
    self.current_step = 0
    state = np.zeros(768, dtype=np.float32)  # example

    return state, {}

  #The point of this function is to send an action to the game
  # and return the observation, reward, terminated, truncated, info
  def step(self, action) -> StepResult:
    self.current_step += 1
    print("ENV STEP CALLED: " + str(self.current_step))
    action = Utils.remove_invalid_actions(action)

    # get reward and state
    obs = self.controller.step(action)

    # #TODO - send action to the server thread 


    # terminate if lives = 0
    reward = obs[State.REWARD]
    terminated = obs[State.LIVES] == 0 and self.current_step > 1
    truncated = self.current_step >= self.max_steps
    if(terminated):
      print("TERMINATED")
      print(obs)
      #reset game
      self.controller.reset()

    if(truncated):
      print("TRUNCATED")
      self.controller.reset()
    info = {}
    return StepResult(obs, reward, terminated, truncated, info)
    #return self.controller.step(action)

  def render(self, mode='human'):
    pass

def my_check_env():
  env = gym.make("MyGame-v0", render_mode=None)
  check_env(env.unwrapped)

class CustomFlatExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
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
    
if __name__ == '__main__':
  #my_check_env()
  policy_kwargs = dict(
    features_extractor_class=CustomFlatExtractor,
    features_extractor_kwargs=dict(features_dim=256)
  )
  print("env run")
  gameEnv = GameEnv(render_mode=True)
  controller = gameEnv.controller 
  #model = PPO.load("models/model_0", env=gameEnv, device="cpu")
  model = PPO.load("models/model_gpu/model_4", env=gameEnv, device="cpu", 
    custom_objects={
        "policy_kwargs": dict(
            features_extractor_class=CustomFlatExtractor
        )
    })

  #Agents.random_agent_process(controller, gameEnv)
  Agents.ppo_trained_model(controller, gameEnv, model)


