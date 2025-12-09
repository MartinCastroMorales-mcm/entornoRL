import subprocess
import time
import os
from enum import IntEnum
import threading
import numpy as np
import game_controller_tcp

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

def test_control():
    # Path to the game executable
    game_path = "./nKaruga"
    
    # Start the game in headless mode with external control
    # -V: Headless
    # -C: External Control
    print("Starting game process...")
    process = subprocess.Popen(
        [game_path, "-C"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0 # Unbuffered
    )
    
    try:
        # Give it a moment to initialize
        time.sleep(1)
        
        if process.poll() is not None:
            print("Game process exited prematurely.")
            stdout, stderr = process.communicate()
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return

        print("Game process running. Sending inputs...")
        
        # Send some inputs
        # 16 is bit 4 (Fire key)
        inputs = [
            0,
            Keys.ENTER, 
            0, 
            0, 
            0, 
            Keys.ENTER, 
            0,
            0,
            0,
            Keys.FIRE, 
            Keys.PAUSE, 
            0, 
            0,
            0,
            Keys.PAUSE, 
            Keys.FIRE
        ]
        
        for i in inputs:
            print(f"Sending input: {i} ({i.name if isinstance(i, Keys) else 'WAIT'})")
            process.stdin.write(f"{int(i)}\n")
            process.stdin.flush()
            time.sleep(0.5)
            
            if process.poll() is not None:
                print("Game crashed or exited.")
                break
        
        print("Test finished. Terminating process.")
        process.terminate()
        process.wait()
        print("Process terminated.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        if process.poll() is None:
            process.terminate()

if __name__ == "__main__":
    test_control()

import subprocess
import time
import os
from enum import IntEnum
#class Keys(IntEnum):
  #DOWN = 1 << 0
  #LEFT = 1 << 1
  #RIGHT = 1 << 2
  #UP = 1 << 3
  #FIRE = 1 << 4
  #POLARITY = 1 << 5
  #FRAGMENT = 1 << 6
  #ESCAPE = 1 << 7
  #PAUSE = 1 << 8
  #RETURN = 1 << 9

class GameController:
    def __init__(self, game_path="./nKaruga", headless=False):
        print("create server")
        self.server = game_controller_tcp.GameControllerTCPServer()
        thread = threading.Thread(target=self.server.start, daemon=True)
        thread.start()
        self.game_path = game_path
        self.headless = headless
        self.process = None
        self.frame_duration = 1.0 / 60.0

    def start(self):
        args = [self.game_path, "-C"]
        if self.headless:
            args.append("-V")
            
        print(f"Starting game: {' '.join(args)}")
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0 # Unbuffered
        )
        # Give it a moment to initialize
        time.sleep(1.0)

    def step(self, action, frames=1) -> np.ndarray:
        """
        Sends an action and waits for the specified number of frames.
        """
        mask = 0
        print("action")
        print(action)
        for i, bit in enumerate(action):
          if bit:
            mask |= (1 << i)

        if self.process is None or self.process.poll() is not None:
            print("Game is not running.")
            exit(1)

        try:
            # Send the action
            self.process.stdin.write(f"{int(mask)}\n")
            self.process.stdin.flush()
            
            # Wait for the duration of the frames
            time.sleep(self.frame_duration * frames)

            # Get new state from your game
            obs = self._get_observation()         # must be shape (608,), float32

            # Compute reward
            reward = self._get_reward()

            # Episode logic
            terminated = False
            truncated = False  # or your own timeout logic

            # Extra debug/info
            info = {}

            #print(f"obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")

            return obs, reward, terminated, truncated, info
            
        except BrokenPipeError:
            print("Game process closed unexpectedly.")

    def getMessage(self):
        return self.server.get_message(None)

    def _get_observation(self):
        return np.zeros(608, dtype=np.float32)

    def _get_reward(self):
        return 0.0


    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Game process terminated.")