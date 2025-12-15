import subprocess
import time
import os
from enum import IntEnum
import threading
import numpy as np

#import game_controller_tcp as game_controller_tcp
from game_controller_tcp import GameControllerTCPServer
from my_types import Keys, State


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
from queue import Queue
from threading import Event


class GameController:
    def __init__(self, game_path="./nKaruga", headless=False):
        print("create server controller")

        self.inbound_queue = Queue()
        self.step_allowed = Event()

        self.server = GameControllerTCPServer(self.inbound_queue, 
                                              self.step_allowed)

        thread = threading.Thread(target=self.server.start, daemon=True)
        thread.start()

        self.game_path = game_path
        self.headless = headless
        self.process = None
        #self.frame_duration = 1.0 / 60.0

    def start(self):
        args = [self.game_path, "-C"]
        if self.headless:
            print("start headless")
            args.append("-V")
        else:
            print("do not start headless")
            
        self.stderr_file = open("game_stderr.log", "w")

        print(f"Starting game: {' '.join(args)}")
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            bufsize=0 # Unbuffered
        )
        # Give it a moment to initialize
        print("game process started")
        time.sleep(2.0)
    
    def reset(self):
        print("Resetting game...")
        # Terminate existing process if running
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Force killing game process")
                self.process.kill()
                self.process.wait()
    
        # Close previous stderr file if open
        if hasattr(self, "stderr_file") and not self.stderr_file.closed:
            self.stderr_file.close()
    
        # Start a fresh game process
        self.start()


    #The point of this function is to get the state of the game
    def step(self, action, frames=1) -> np.ndarray:
        """
        Sends an action and waits for the specified number of frames.
        """
        mask = 0
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
            
            msg = self.inbound_queue.get()
            self.step_allowed.set()      
            obs = np.fromstring(msg, sep=",", dtype=np.float32)
            #obs fix
            state = obs[:State.CHAIN_COLOR]      # frame .. chain_color
            enemy_bullet_data = obs[State.ENEMIES:]     # e_0_x .. end

            obs = np.concatenate((state, enemy_bullet_data))

            return obs
            
        except BrokenPipeError:
            print("Game process closed unexpectedly.")

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Game process terminated.")