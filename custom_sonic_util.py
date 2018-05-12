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

class CustomSonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(CustomSonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['RIGHT', 'DOWN'], ['B']]

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
        self._prev_rew  = 0
        self._continous_zero_rew_time = 0
        self._continous_plus_rew_time = 0
        self._start_time = datetime.now()
        
    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        self._prev_rew  = 0
        self._continous_zero_rew_time = 0
        self._continous_plus_rew_time = 0
        self._start_time = datetime.now()
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        #print('Init ',rew, action)
   
        # Allow Backtracking
        backTracked = False
        backTrackAbsRew = 0

        if (rew < 0):
            backTracked = True
            backTrackAbsRew = abs(rew)

        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
	    #print('Allow Backtracking ',rew, backTrackAbsRew)
        
        # rew speed
        deltaRew = rew - self._prev_rew
        self._prev_rew = rew
        if (deltaRew > 3):
            #print (deltaRew)
            rew *= deltaRew

        # escape from stuck
        if (((backTracked and (backTrackAbsRew < 1))) or \
            (rew >= 0 and rew < 1)):
            if (self._continous_zero_rew_time < 10):
                rew -= 3
                self._continous_zero_rew_time += 1
                self._continous_plus_rew_time = 0
                #print('stuck %d %d' % (self._continous_zero_rew_time, rew))
            else:
                rew -= 100
                self._continous_zero_rew_time = 0
                self._continous_plus_rew_time = 0
                #print('failed from escape %d %d' % (self._continous_zero_rew_time, rew))
        else:
               #print('SUCCESSED escape %d %d %d' % (self._continous_plus_rew_time, self._continous_zero_rew_time, rew))
               self._continous_zero_rew_time = 0
               self._continous_plus_rew_time += 1
               rew += self._continous_plus_rew_time

        # live long
        if (done):
           delta = datetime.now() -  self._start_time
           if (delta.seconds <= 120):
                rew -= 100
           else:
                rew += 30
           #print('live long %d %d' % (delta.seconds, rew))

        return obs, rew, done, info