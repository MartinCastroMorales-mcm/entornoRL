import time
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
from enum import IntEnum
import numpy as np

from v0_game_hooks import GameController
from v0_game_hooks import Keys

register(
  id='MyGame-v0',
  entry_point='v0_game_env:GameEnv',
)

class GameEnv(gym.Env):
  metadata = {'render_modes': ['human'], 'render_fps': 1}



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

  def keys_to_actions(self, keys_mask):
    # 10 actions total
    actions = [0] * len(self.Action)

    if keys_mask & Keys.UP:
        actions[self.Action.UP] = 1
    if keys_mask & Keys.DOWN:
        actions[self.Action.DOWN] = 1
    if keys_mask & Keys.LEFT:
        actions[self.Action.LEFT] = 1
    if keys_mask & Keys.RIGHT:
        actions[self.Action.RIGHT] = 1
    if keys_mask & Keys.FIRE:
        actions[self.Action.FIRE] = 1
    if keys_mask & Keys.POLARITY:
        actions[self.Action.POLARITY] = 1
    if keys_mask & Keys.FRAGMENT:
        actions[self.Action.FRAGMENT] = 1
    if keys_mask & Keys.ESCAPE:
        actions[self.Action.ESCAPE] = 1
    if keys_mask & Keys.PAUSE:
        actions[self.Action.PAUSE] = 1
    if keys_mask & Keys.ENTER:
        actions[self.Action.ENTER] = 1

    return actions



  class State(IntEnum):
    FRAME      = 0
    SCORE      = 1
    LIVES      = 2
    POWER      = 3
    POLARITY   = 4
    PLAYER_X   = 5
    PLAYER_Y   = 6
    CHAIN      = 7

    RESERVED_START = 8
    RESERVED_END   = 19

    ENEMIES    = 19 
    BULLETS    = 179

  def __init__(self, render_mode=None):
    self.reset()
    self.renderMode = render_mode
    if(self.renderMode):
      controller = GameController(headless=False)
    else:
      controller = GameController(headless=True)


    controller.start()
    self.controller = controller

    # action space
    self.action_space = spaces.MultiBinary(len(self.Action))

    # observation space
    # TODO - define more precise observation space
    self.observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(608,),
    dtype=np.float32
    )
  
  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    # initialize value - i think i dont have these
    state = np.zeros(608, dtype=np.float32)  # example

    return state, {}


  def step(self, action):
    reward = 0
    if(action[7] == 1):
      action[7] = 0
    if(action[8] == 1):
      action[8] = 0
    if(action[GameEnv.Action.RIGHT] == 1 and action[GameEnv.Action.LEFT] == 1):
      action[GameEnv.Action.RIGHT] = 0
      action[GameEnv.Action.LEFT] = 0
    if(action[GameEnv.Action.DOWN] == 1 and action[GameEnv.Action.UP] == 1):
      action[GameEnv.Action.DOWN] = 0
      action[GameEnv.Action.UP] = 0
    if(action[GameEnv.Action.ENTER] == 1):
      action[GameEnv.Action.ENTER] = 0
    return self.controller.step(action)

  def readInput(self):
      return self.server.get_message(None)

  def render(self, mode='human'):
    pass

def my_check_env():
  from gymnasium.utils.env_checker import check_env
  env = gym.make("MyGame-v0", render_mode=None)
  check_env(env.unwrapped)


def random_agent(env):
   return env.action_space.sample()

def random_agent_process(controller):
  while True:
     #read input
     input = controller.getMessage()
     print(f"Input: {input}")
     # do something with the input

     action = random_agent(gameEnv)
     action = action.tolist()
     print(f"Random Agent: {action}")
     print(f"Random Agent: {gameEnv.keys_to_actions(0)}")
     if(action[7] == 1):
        print("ESCAPE")
        action[7] = 0
     if(action[8] == 1):
        print("PAUSE")
        action[8] = 0
     if(action[GameEnv.Action.RIGHT] == 1 and action[GameEnv.Action.LEFT] == 1):
       print("FIRE")
       action[GameEnv.Action.RIGHT] = 0
       action[GameEnv.Action.LEFT] = 0
     if(action[GameEnv.Action.DOWN] == 1 and action[GameEnv.Action.UP] == 1):
       print("FIRE")
       action[GameEnv.Action.DOWN] = 0
       action[GameEnv.Action.UP] = 0
     if(action[GameEnv.Action.ENTER] == 1):
        print("ENTER")
        action[GameEnv.Action.ENTER] = 0
     controller.step(action, frames=10)
     controller.step(gameEnv.keys_to_actions(0), frames=10)
     controller.step(gameEnv.keys_to_actions(0), frames=10)
     controller.step(gameEnv.keys_to_actions(0), frames=10)
     controller.step(gameEnv.keys_to_actions(0), frames=10)

if __name__ == '__main__':
  #my_check_env()
  # 1. Navigate Menu

  gameEnv = GameEnv()
  controller = gameEnv.controller
  #print("Navigating Menu...")

  #controller.step(gameEnv.keys_to_actions(Keys.ENTER), frames=10) # Open Menu
  #controller.step(gameEnv.keys_to_actions(0), frames=60)           # Wait 1s
  #controller.step(gameEnv.keys_to_actions(Keys.ENTER), frames=10) # Start Game
  #controller.step(gameEnv.keys_to_actions(0), frames=120)          # Wait 2s for game to load
  #print("Restarting ...")
        
  ## 2. Move Right for 1 second (60 frames)
  ## We send the command every 10 frames, so 6 steps
  #print("Moving Right...")
  #for _ in range(6):
      #controller.step(gameEnv.keys_to_actions(Keys.RIGHT), frames=10)
            
  ## 3. Stop for 0.5 second
  #print("Stopping...")
  #controller.step(gameEnv.keys_to_actions(0), frames=30)
        
  ## 4. Move Left for 1 second
  #print("Moving Left...")
  #for _ in range(6):
      #controller.step(gameEnv.keys_to_actions(Keys.LEFT), frames=10)
            
  ## 5. Fire for 1 second
  #print("Firing...")
  #for _ in range(6):
      #controller.step(gameEnv.keys_to_actions(Keys.FIRE), frames=10)

  ##print("Moving Left...")
  ##for _ in range(1000):
    ##for _ in range(6):
        ##controller.step(gameEnv.keys_to_actions(Keys.LEFT), frames=10)
    ##for _ in range(6):
        ##controller.step(gameEnv.keys_to_actions(Keys.RIGHT), frames=10)

  #print("Test Complete.")
  random_agent_process(controller)


