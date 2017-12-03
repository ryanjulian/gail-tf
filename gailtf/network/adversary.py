from gailtf.baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow as tf
from gailtf.baselines.common import tf_util as U
from gailtf.common.tf_util import *
import numpy as np
import ipdb

"""This is the discriminator that distinguishes fake from real trajectories."""


class TransitionClassifier(object):
    def __init__(self, observation_shape, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="adversary"):
        self.scope = scope
        self.observation_shape = observation_shape
        self.actions_shape = np.array([0, 0])  # env.action_space.shape
        self.input_shape = self.observation_shape
        self.hidden_size = hidden_size
        self.build_ph()
        # Build graph
        generator_logits = self.build_graph(self.generator_obs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function(
            [self.generator_obs_ph, self.expert_obs_ph],
            self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="observations_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="expert_observations_ph")

    def build_graph(self, obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean / self.obs_rms.std)
            _input = obs  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs):
        sess = U.get_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)

        feed_dict = {self.generator_obs_ph: obs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward
