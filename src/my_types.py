from typing import NamedTuple, Dict, Any
from enum import IntEnum
import numpy as np


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