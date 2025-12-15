from v0_game_hooks import GameController
from my_types import Action, State
from utils import Utils
from gymnasium import Env
from stable_baselines3 import PPO


class Agents:

  def random_agent_process(controller: GameController, env: Env):
    while True:
      #read input
      input = controller.getMessage()
      print(f"Input: {input}")
      # do something with the input

      action = env.action_space.sample()
      action = action.tolist()
      print(f"Random Agent: {action}")
      print(f"Random Agent: {Utils.keys_to_actions(0)}")
      if(action[7] == 1):
          print("ESCAPE")
          action[7] = 0
      if(action[8] == 1):
          print("PAUSE")
          action[8] = 0
      if(action[Action.RIGHT] == 1 and action[Action.LEFT] == 1):
        print("FIRE")
        action[Action.RIGHT] = 0
        action[Action.LEFT] = 0
      if(action[Action.DOWN] == 1 and action[Action.UP] == 1):
        print("FIRE")
        action[Action.DOWN] = 0
        action[Action.UP] = 0
      if(action[Action.ENTER] == 1):
          print("ENTER")
          action[Action.ENTER] = 0
      controller.step(action, frames=10)
      controller.step(Utils.keys_to_actions(0), frames=10)
      controller.step(Utils.keys_to_actions(0), frames=10)
      controller.step(Utils.keys_to_actions(0), frames=10)
      controller.step(Utils.keys_to_actions(0), frames=10)

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

        