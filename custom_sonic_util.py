"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np

from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym_remote.client as grc
import time
from datetime import datetime

def make_env(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv('tmp/sock')
    env = CustomSonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class CustomSonicDiscretizer(SonicDiscretizer):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                  ['DOWN', 'B'], ['B']]

        #actions = [['LEFT'] ,['RIGHT'] , ['RIGHT', 'DOWN'], ['B']]

        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()
    
class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class RewardPolicy(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(RewardPolicy, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self._continous_zero_rew_time = 0
        self._start_time = datetime.now()
        
    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        self._continous_zero_rew_time = 0
        self._start_time = datetime.now()
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
   
        #Encourage Long distance
        if (rew > 10):
            rew *= 3

        # Allow Backtracking
        backTracked = False
        backTrackAbsRew = 0

        if (rew < 0):
            #print('Allow Backtracking %d' %rew)
            backTracked = True
            backTrackAbsRew = abs(rew)
            self._cur_x += rew
            rew = max(0, self._cur_x - self._max_x)
            self._max_x = max(self._max_x, self._cur_x)
        
        # escape from stuck
        if (((backTracked and (backTrackAbsRew < 1))) or \
            (rew >= 0 and rew < 1)):
            if (self._continous_zero_rew_time < 10):
                rew -= self._continous_zero_rew_time*10
                self._continous_zero_rew_time += 1
               # print('escape from stuck %d %d' % (self._continous_zero_rew_time, rew))
            else:
                self._continous_zero_rew_time = 0
        else:
            self._continous_zero_rew_time = 0

        # live long
        if (done):
           delta = datetime.now() -  self._start_time
           if (delta.seconds <= 180):
                rew -= delta.seconds
           else:
                rew += delta.seconds
           #print('live long %d %d' % (delta.seconds, rew))

        return obs, rew, done, info