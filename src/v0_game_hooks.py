import subprocess
import time
import os
from queue import Queue
from threading import Event
import threading

import numpy as np

from game_controller_tcp import GameControllerTCPServer
from my_types import State
#from v0_game_controller_tcp import GameControllerTCPServer


class GameController:
    def __init__(self, game_path="./nKaruga", headless=False):
        print("create server controller")

        self.inbound_queue = Queue()
        self.step_allowed = Event()

        self.server = GameControllerTCPServer(self.inbound_queue,
                                              self.step_allowed)

        thread = threading.Thread(target=self.server.start, daemon=True)
        thread.start()

        self.server.ready.wait()

        self.game_path = game_path
        self.headless = headless
        self.process = None
        #self.frame_duration = 1.0 / 60.0

    def start(self):
        port = self.server.server.getsockname()[1]
        args = [self.game_path, f"-C -P {port}"]
        if self.headless:
            print("start headless")
            args.append("-V")
            args.append("-S")
            #args.append("-U")
        else:
            print("do not start headless")

        self.stderr_file = open("game_stderr.log", "w")

        my_env = os.environ.copy()
        # Prepend /bundle/lib to LD_LIBRARY_PATH. Use ':' to separate paths.
        # Ensure existing LD_LIBRARY_PATH is included if it exists.
        #if 'LD_LIBRARY_PATH' in my_env:
        #    my_env['LD_LIBRARY_PATH'] = '/content/bundle/lib:' + my_env['LD_LIBRARY_PATH']
        #else:
        #    my_env['LD_LIBRARY_PATH'] = '/content/bundle/lib'

        print(f"Starting game: {' '.join(args)}")
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            bufsize=0, # Unbuffered
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
            msg = None
            obs = None
            # Send the action
            self.process.stdin.write(f"{int(mask)}\n")
            self.process.stdin.flush()

            #print("msg")
            #log("log")
            #if self.inbound_queue.empty():
            #    print("inbound queue empty")
            #    #print(self.server.currentState)
            #    #self.step_allowed.wait()
            #    msg = np.zeros(768, dtype=np.float32) 
            #    msg[State.LIVES] = 4
            #    self.step_allowed.set()
            #    #print(msg)
            #    obs = msg
            #else:
            
            msg = self.inbound_queue.get()
            self.step_allowed.set()
            
            print("inbound queue")
            self.step_allowed.set()
            #print("controller step set")
            obs = np.fromstring(msg, sep=",", dtype=np.float32)
            #print(f"obs: {obs}")
            #obs fix
            state = obs[:State.CHAIN_COLOR]      # frame .. chain_color
            enemy_bullet_data = obs[State.ENEMIES:]     # e_0_x .. end
            obs = np.concatenate((state, enemy_bullet_data))

            return obs

        except BrokenPipeError:
            print("Game process closed unexpectedly.")
            raise BrokenPipeError
            

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Game process terminated.")