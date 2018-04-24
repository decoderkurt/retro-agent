#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BasicPlayer,BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre
from retro_contest.local import make
##from sonic_util import AllowBacktracking, make_env
import time

class SimplePlayer(BasicPlayer):
    def _gather_transition(self):
        if self._needs_reset:
            self._needs_reset = False
            self._cur_state = self.model.start_state(1)
            self._last_obs = self.env.reset()
            self._episode_id += 1
            self._episode_step = 0
            self._total_reward = 0.0
        output = self.model.step([self._last_obs], self._cur_state)
        new_obs, rew, self._needs_reset, info = self.env.step(env.action_space.sample())
        self._total_reward += rew
        res = {
            'obs': self._last_obs,
            'model_outs': output,
            'rewards': [rew],
            'new_obs': (new_obs if not self._needs_reset else None),
            'info': info,
            'start_state': self._cur_state,
            'episode_id': self._episode_id,
            'episode_step': self._episode_step,
            'end_time': time.time(),
            'is_last': self._needs_reset,
            'total_reward': self._total_reward
        }
        self._cur_state = output['states']
        self._last_obs = new_obs
        self._episode_step += 1
        return res

class RainbowDQN(DQN):
    def train(self,
              env,
              num_steps,
              player,
              replay_buffer,
              optimize_op,
              train_interval=1,
              target_interval=8192,
              batch_size=32,
              min_buffer_size=20000,
              tf_schedules=(),
              handle_ep=lambda steps, rew: None,
              timeout=None):       
        sess = self.online_net.session
        sess.run(self.update_target)
        steps_taken = 0
        next_target_update = target_interval
        next_train_step = train_interval
        start_time = time.time()
        obs = env.reset()
        while steps_taken < num_steps:
            if timeout is not None and time.time() - start_time > timeout:
                return
            env.render()
            transitions = player.play()
            for trans in transitions:
                if trans['is_last']:
                    handle_ep(trans['episode_step'] + 1, trans['total_reward'])
                replay_buffer.add_sample(trans)
                steps_taken += 1
                for sched in tf_schedules:
                    sched.add_time(sess, 1)
                if replay_buffer.size >= min_buffer_size and steps_taken >= next_train_step:
                    next_train_step = steps_taken + train_interval
                    batch = replay_buffer.sample(batch_size)
                    _, losses = sess.run((optimize_op, self.losses),
                                         feed_dict=self.feed_dict(batch))
                    replay_buffer.update_weights(batch, losses)
                if steps_taken >= next_target_update:
                    next_target_update = steps_taken + target_interval
                    sess.run(self.update_target)

def main():
    """Run DQN until the environment throws an exception."""
    ## env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    ## env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = RainbowDQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        ##player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        player = SimplePlayer(env, dqn.online_net, 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())
        dqn.train(env,
                  num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)

