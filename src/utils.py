from my_types import Action, Keys
import joblib
import torch as th
import zipfile

scaler = joblib.load('scaler.pkl')

class Utils:


# Before predicting:
  def normalize_observation(obs):
    # obs is a numpy array of shape (1, 778) #tabla
    obs_reshaped = obs.reshape(1, -1) 
    #normaliza
    return scaler.transform(obs_reshaped)[0] #[0] para que sea (778,)
  #static
  def remove_invalid_actions(action):
    if(action[7] == 1):
      action[7] = 0
    if(action[8] == 1):
      action[8] = 0
    if(action[Action.RIGHT] == 1 and action[Action.LEFT] == 1):
      action[Action.RIGHT] = 0
      action[Action.LEFT] = 0
    if(action[Action.DOWN] == 1 and action[Action.UP] == 1):
      action[Action.DOWN] = 0
      action[Action.UP] = 0
    if(action[Action.ENTER] == 1):
      action[Action.ENTER] = 0
    return action

  
  def keys_to_actions(keys_mask):
    # 10 actions total
    actions = [0] * len(Action)

    if keys_mask & Keys.UP:
        actions[Action.UP] = 1
    if keys_mask & Keys.DOWN:
        actions[Action.DOWN] = 1
    if keys_mask & Keys.LEFT:
        actions[Action.LEFT] = 1
    if keys_mask & Keys.RIGHT:
        actions[Action.RIGHT] = 1
    if keys_mask & Keys.FIRE:
        actions[Action.FIRE] = 1
    if keys_mask & Keys.POLARITY:
        actions[Action.POLARITY] = 1
    if keys_mask & Keys.FRAGMENT:
        actions[Action.FRAGMENT] = 1
    if keys_mask & Keys.ESCAPE:
        actions[Action.ESCAPE] = 1
    if keys_mask & Keys.PAUSE:
        actions[Action.PAUSE] = 1
    if keys_mask & Keys.ENTER:
        actions[Action.ENTER] = 1

    return actions


  def load_zip(zip_path, env):
    with zipfile.ZipFile("models/model_20_000_steps.zip", "r") as archive:
        with archive.open("policy.pth") as weights_file:
            state_dict = th.load(weights_file, map_location="cpu")
    return state_dict