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
from my_types import NeuralNetwork, StepResult, State, Action, NEURONAS_ENTRANTES
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
    shape=(NEURONAS_ENTRANTES,),
    dtype=np.float32
    )
  
  def reset(self, seed=None, options=None):
    print("ENV RESET CALLED")
    super().reset(seed=seed)
    # initialize value - i think i dont have these
    self.current_step = 0
    state = np.zeros(NEURONAS_ENTRANTES, dtype=np.float32)  # example

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

def transfer_weights(bc_model, ppo):
  with th.no_grad():

      # BC layers
      bc_layers = list(bc_model.linear_relu_stack)

      # PPO actor layers
      ppo_policy_net = ppo.policy.mlp_extractor.policy_net
      ppo_action_net = ppo.policy.action_net

      # Copy hidden layers
      ppo_policy_net[0].weight.data.copy_(bc_layers[0].weight.data)
      ppo_policy_net[0].bias.data.copy_(bc_layers[0].bias.data)

      ppo_policy_net[2].weight.data.copy_(bc_layers[2].weight.data)
      ppo_policy_net[2].bias.data.copy_(bc_layers[2].bias.data)

      # Copy output layer
      # Copy only the first 6 actions
      ppo_action_net.weight[:6].copy_(bc_layers[4].weight)
      ppo_action_net.bias[:6].copy_(bc_layers[4].bias)

      # Copy only the first 6 actions
      ppo_action_net.weight[:6].copy_(bc_layers[4].weight)
      ppo_action_net.bias[:6].copy_(bc_layers[4].bias)

      # Initialize remaining 4 actions safely
      ppo_action_net.weight[6:].zero_()
      ppo_action_net.bias[6:].zero_()
  

if __name__ == '__main__':
  #my_check_env()
  print("Seleccione un modelo")
  print("1: Random")
  print("2: BC")
  print("3: BC con transferencia de pesos a PPO")
  print("4: PPO simple")
  print("5: PPO simple")
  print("6: PPO 10 000")
  print("7: PPO 10 000 + bc")
  input = input()

  gameEnv = GameEnv(render_mode=True)
  controller = gameEnv.controller 
  match input:
    case "1":
      print("random")
      Agents.random_agent_process(controller, gameEnv)
    #case "1":
      #print("PPO")
      #model = PPO.load("models/model_0", env=gameEnv, device="cpu")
    case "2":
      print("BC")
      state_dict = th.load("models/model.pth", map_location="cpu")
      bc_model = NeuralNetwork()
      bc_model.load_state_dict(state_dict)
      Agents.bc_trained_model(controller, gameEnv, bc_model)
    case "3":
      print("BC con transferencia de pesos a PPO")
      policy_kwargs = dict(
        #Se usa la misma arquitectura que en el entrenamiento de pytorch
        net_arch=[512, 512],
        activation_fn=nn.ReLU
      )
      # Se crea un algoritmo de PPO con la arquitectura de la 
      # red de pytorch
      ppo = PPO(
          policy="MlpPolicy",
          env=gameEnv,
          policy_kwargs=policy_kwargs,
          device="cpu"
      )
      state_dict = th.load("models/model.pth", map_location="cpu")
      bc_model = NeuralNetwork()
      bc_model.load_state_dict(state_dict)
      transfer_weights(bc_model, ppo)
      # Initialize remaining 4 actions safely
      ppo_action_net = ppo.policy.action_net
      bc_layers = list(bc_model.linear_relu_stack)
      ppo_action_net.weight[6:].zero_()
      ppo_action_net.bias[6:].zero_()
      ppo_action_net.weight.data.copy_(bc_layers[4].weight.data)
      ppo_action_net.bias.data.copy_(bc_layers[4].bias.data)
      Agents.ppo_trained_model(controller, gameEnv, ppo)
    case "4":
      print("PPO simple 4")
      model = PPO.load("models/model_0", env=gameEnv, device="cpu")
      Agents.ppo_trained_model(controller, gameEnv, model)
    case "5":
      print("PPO simple 5")
      model = PPO.load("models/model_gpu/model_4", env=gameEnv, device="cpu")
      Agents.ppo_trained_model(controller, gameEnv, model)
    case _:
      print("Invalid option")
      exit(1)


  #policy_kwargs = dict(
    #features_extractor_class=CustomFlatExtractor,
    #features_extractor_kwargs=dict(features_dim=256)
  #)
  #print("env run")
  #model = PPO.load("models/model_0", env=gameEnv, device="cpu")
  #model = PPO.load("models/model_gpu/model_4", env=gameEnv, device="cpu", 
    #custom_objects={
        #"policy_kwargs": dict(
            #features_extractor_class=CustomFlatExtractor
        #)
    #})





  #transfer_weights(bc_model, ppo)

  # copy weights from bc_model to ppo
  
  

  #Agents.random_agent_process(controller, gameEnv)
  #Agents.ppo_trained_model(controller, gameEnv, ppo)
  #Agents.bc_trained_model(controller, gameEnv, bc_model)


