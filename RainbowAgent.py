#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf

import time
from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.rollouts.rollers import _reduce_states, _inject_state, _reduce_model_outs
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre
from custom_sonic_util import RewardPolicy, make_env


class RainbowBatchedPlayer(BatchedPlayer):
     def _step_sub_batch(self, sub_batch):
        model_outs = self.model.step(self._last_obses[sub_batch], self._cur_states[sub_batch])
        self.batched_env.step_start(model_outs['actions'], sub_batch=sub_batch)
        outs = self.batched_env.step_wait(sub_batch=sub_batch)
        end_time = time.time()
        transitions = []
        for i, (obs, rew, done, info) in enumerate(zip(*outs)):
            self._total_rewards[sub_batch][i] += rew
            #print(obs, ' ', rew, ' ', done, ' ',info)
           # print(rew, ' ', done, ' ',info)
            transitions.append({
                'obs': self._last_obses[sub_batch][i],
                'model_outs': _reduce_model_outs(model_outs, i),
                'rewards': [rew],
                'new_obs': (obs if not done else None),
                'info': info,
                'start_state': _reduce_states(self._cur_states[sub_batch], i),
                'episode_id': self._episode_ids[sub_batch][i],
                'episode_step': self._episode_steps[sub_batch][i],
                'end_time': end_time,
                'is_last': done,
                'total_reward': self._total_rewards[sub_batch][i]
            })
            if done:
                _inject_state(model_outs['states'], self.model.start_state(1), i)
                self._episode_ids[sub_batch][i] = self._next_episode_id
                self._next_episode_id += 1
                self._episode_steps[sub_batch][i] = 0
                self._total_rewards[sub_batch][i] = 0.0
            else:
                self._episode_steps[sub_batch][i] += 1
        self._cur_states[sub_batch] = model_outs['states']
        self._last_obses[sub_batch] = outs[0]
        return transitions

class RainbowPlayer(NStepPlayer):
    def _play_once(self):
        for trans in self.player.play():
        #    if len(trans['info']) > 0:
        #        print(trans['info'])
            assert len(trans['rewards']) == 1
            ep_id = trans['episode_id']
            if ep_id in self._ep_to_history:
                self._ep_to_history[ep_id].append(trans)
            else:
                self._ep_to_history[ep_id] = [trans]
        res = []
        for ep_id, history in list(self._ep_to_history.items()):
            while history:
               # print(ep_id, ': history ' ,history)
                trans = self._next_transition(history)
                #print(ep_id, ': trans info' ,trans['info'])
                if trans is None:
                    break
                res.append(trans)
            if not history:
                del self._ep_to_history[ep_id]
        return res

    def _next_transition(self, history):
        if len(history) < self.num_steps:
            if not history[-1]['is_last']:
                return None
        res = history[0].copy()
        res['rewards'] = [h['rewards'][0] for h in history[:self.num_steps]]
        res['total_reward'] += sum(h['rewards'][0] for h in history[1:self.num_steps])
        if len(history) >= self.num_steps:
            res['new_obs'] = history[self.num_steps-1]['new_obs']
        else:
            res['new_obs'] = None
        del history[0]
        return res

class RainbowDQN(DQN):
    def __init__(self, online_net, target_net, discount=0.9999):

        self.online_net = online_net
        self.target_net = target_net
        self.discount = discount

        obs_shape = (None,) + online_net.obs_vectorizer.out_shape
        self.obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.actions_ph = tf.placeholder(tf.int32, shape=(None,))
        self.rews_ph = tf.placeholder(tf.float32, shape=(None,))
        self.new_obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.terminals_ph = tf.placeholder(tf.bool, shape=(None,))
        self.discounts_ph = tf.placeholder(tf.float32, shape=(None,))
        self.weights_ph = tf.placeholder(tf.float32, shape=(None,))

        losses = online_net.transition_loss(target_net, self.obses_ph, self.actions_ph,
                                            self.rews_ph, self.new_obses_ph, self.terminals_ph,
                                            self.discounts_ph)
        self.losses = self.weights_ph * losses
        self.loss = tf.reduce_mean(self.losses)

        assigns = []
        for dst, src in zip(target_net.variables, online_net.variables):
            assigns.append(tf.assign(dst, src))
        self.update_target = tf.group(*assigns)

     # pylint: disable=R0913,R0914
    def train(self,
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
        
        #saver
        #saver = tf.train.Saver()

        steps_taken = 0
        next_target_update = target_interval
        next_train_step = train_interval
        start_time = time.time()
        while steps_taken < num_steps:
            if timeout is not None and time.time() - start_time > timeout:
                return
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
                #save
                #ckpt_path = saver.save(sess, "/root/compo/rainbow_agent.ckpt")
                #print("ckpt saved as : ", ckpt_path)

def main():
    """Run DQN until the environment throws an exception."""
    env = RewardPolicy(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    


    with tf.Session(config=config) as sess:
        dqn = RainbowDQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = RainbowPlayer(RainbowBatchedPlayer(env, dqn.online_net), 4)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.7, 0.7, epsilon=0.1),
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
