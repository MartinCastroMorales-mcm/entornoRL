from v0_game_hooks import GameController
from my_types import Action, State
from utils import Utils
from gymnasium import Env
from stable_baselines3 import PPO
import torch as th
import numpy as np


class Agents:


  def random_action_generator() -> np.float32:
    return np.random.randint(0, 2, size=10)

  def random_agent_process(controller: GameController, env: Env):
    done = False
    while not done:
      #read input
      #input = Agents.random_action_generator()
      #print(f"Input: {input}")
      # do something with the input

      action = env.action_space.sample()
      action = action.tolist()
      print(f"Random Agent: {action}")
      print(f"Random Agent: {Utils.keys_to_actions(0)}")
      action = Utils.remove_invalid_actions(action)
      obs = controller.step(action, frames=10)
      #controller.step(Utils.keys_to_actions(0), frames=10)
      #controller.step(Utils.keys_to_actions(0), frames=10)
      #controller.step(Utils.keys_to_actions(0), frames=10)
      #controller.step(Utils.keys_to_actions(0), frames=10)

      terminated = obs[State.LIVES] == 0 
      if(terminated):
        print("TERMINATED")
        done = True

  def ppo_trained_model(controller: GameController, eval_env: Env, model: PPO):
    (obs, info) = eval_env.reset()
    done = False
    while not done:
      action, _ = model.predict(obs)
      action = Utils.remove_invalid_actions(action)
      obs = controller.step(action)

      #obs, reward, terminated, truncated, info = eval_env.step(action)

      terminated = obs[State.LIVES] == 0 
      if(terminated):
        print("TERMINATED")
        done = True

        
  def torch_policy_step(model, obs):
      obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)  # (1, obs_dim)
      print("obs_tensor")
      print(len(obs_tensor))

      with th.no_grad():
        logits = model(obs_tensor)  # (1, n_actions)
        probs = th.sigmoid(logits)
        action = (probs > 0.5).int().squeeze(0).cpu().numpy()
        print(f"action: {action}")
      
      if(len(action) == 6):
        b = np.zeros(10, dtype=np.float32)
        b[:len(action)] = action

      # OR stochastic (recommended if you want PPO-like behavior)
      # dist = th.distributions.Categorical(logits=logits)
      # action = dist.sample().item()

      return b

  def bc_trained_model(controller: GameController, eval_env: Env, 
                      model: th.nn.Module):
    obs, info = eval_env.reset()
    done = False

    model.eval()

    while not done:
      action = Agents.torch_policy_step(model, obs)
      action = Utils.remove_invalid_actions(action)
      obs = controller.step(action)

      terminated = obs[State.LIVES] == 0
      if terminated:
          print("TERMINATED")
          done = True

