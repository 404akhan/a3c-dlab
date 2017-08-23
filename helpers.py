import numpy as np
import deepmind_lab
from skimage.color import rgb2gray


def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTION_LIST = [
  _action(-20, 0, 0, 0, 0, 0, 0),
  _action(20, 0, 0, 0, 0, 0, 0),
  _action(0, 10, 0, 0, 0, 0, 0),
  _action(0, -10, 0, 0, 0, 0, 0),
  _action(0, 0, -1, 0, 0, 0, 0),
  _action(0, 0, 1, 0, 0, 0, 0),
  _action(0, 0, 0, 1, 0, 0, 0),
  _action(0, 0, 0, -1, 0, 0, 0),
  _action(0, 0, 0, 0, 1, 0, 0),
  _action(0, 0, 0, 0, 0, 1, 0),
  _action(0, 0, 0, 0, 0, 0, 1)
]


class WrapEnv(object):
  def __init__(self, level, width=80, height=80, fps=60):
    self.env = deepmind_lab.Lab(
      level, ['RGB_INTERLACED'],
      config={
          'fps': str(fps),
          'width': str(width),
          'height': str(height)
    })

    self.num_actions = len(ACTION_LIST)

  def reset(self):
    self.env.reset()
    obs = self.env.observations()
    obs = obs['RGB_INTERLACED']
    obs = rgb2gray(obs)
    obs = np.expand_dims(obs, axis=0)

    obs = obs.astype(np.float32)
    obs *= (1.0 / 255.0)

    return obs

  def step(self, action):
    reward = self.env.step(ACTION_LIST[action], num_steps=1)
    done = not self.env.is_running()

    if not done:
      obs = self.env.observations()
      obs = obs['RGB_INTERLACED']
      obs = rgb2gray(obs)
      obs = np.expand_dims(obs, axis=0)
    else:
      obs = np.zeros((1, 80, 80)) # todo check datatype, maybe int8

    obs = obs.astype(np.float32)
    obs *= (1.0 / 255.0)

    return obs, reward, done, {'ale.lives': 0}

