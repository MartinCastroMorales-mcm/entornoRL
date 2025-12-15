import gymnasium as gym
from v0_game_env import GameEnv
from stable_baselines3 import PPO

import os

def train():
  model_dir = "models"
  log_dir = "logs"

  os.makedirs(model_dir, exist_ok=True)
  os.makedirs(log_dir, exist_ok=True)

  env = gym.make("MyGame-v0", render_mode=None)
  #MlpPolicy define una arquitectura de red capas densas,
  #Tambien se puede ocupar CnnPolicy para una con capas 
  #convolucionales, especialmente para imagenes
  model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=log_dir)
  #model.learn(total_timesteps=100000)

  TIMESTEPS = 1
  MAX_ITERS = 1
  iters = 0

  print("Training...")

  for iters in range(MAX_ITERS):
    print(f"Iter {iters}")

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{model_dir}/model_{iters}")

if __name__ == '__main__':
  print("test")
  train()
