"""
Baseline Agents
===============
Three standard approaches for handling plasticity loss in continual RL:

1. ShrinkAndPerturbAgent: Periodically shrink weights toward zero and add noise.
   (Ash & Adams, 2020 -- "Warm-Starting Neural Network Training")

2. PeriodicResetAgent: Fully reset the online network every K episodes.
   Simple but disruptive; serves as an upper bound on adaptation speed.

3. L2RegAgent: Add L2 regularization to keep weights small, preventing
   the weight explosion associated with plasticity loss.
"""

import numpy as np
import tensorflow as tf
from dqn_base import DQNAgent


class ShrinkAndPerturbAgent(DQNAgent):
    """
    Shrink-and-Perturb baseline (Ash & Adams, 2020).
    Every `perturb_freq` steps: w <- alpha * w + noise(0, sigma)
    """

    def __init__(self, *args, perturb_freq=2000, alpha=0.9, sigma=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb_freq = perturb_freq
        self.alpha  = alpha
        self.sigma  = sigma
        self.name   = "ShrinkAndPerturb"

    def train_step(self):
        loss = super().train_step()
        if self.step_count > 0 and self.step_count % self.perturb_freq == 0:
            self._shrink_and_perturb()
        return loss

    def _shrink_and_perturb(self):
        new_weights = []
        for w in self.online_net.get_weights():
            noise = np.random.normal(0, self.sigma, size=w.shape).astype(w.dtype)
            new_weights.append(self.alpha * w + noise)
        self.online_net.set_weights(new_weights)
        # Sync target net after perturbation
        self.target_net.set_weights(self.online_net.get_weights())


class PeriodicResetAgent(DQNAgent):
    """
    Periodic network reset baseline.
    Every `reset_freq` episodes: reinitialise the online network weights.
    Preserves the replay buffer so experience is not lost.
    """

    def __init__(self, *args, reset_freq=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_freq = reset_freq
        self.name = "PeriodicReset"

    def on_episode_end(self, episode_reward: float):
        super().on_episode_end(episode_reward)
        if self.episode_count % self.reset_freq == 0:
            self._reset_network()

    def _reset_network(self):
        from dqn_base import build_qnet
        self.online_net = build_qnet(
            self.obs_dim, self.n_actions, self.hidden_sizes
        )
        self.target_net = build_qnet(
            self.obs_dim, self.n_actions, self.hidden_sizes
        )
        self.target_net.set_weights(self.online_net.get_weights())
        # Reset optimizer state too
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.optimizer.learning_rate
        )


class L2RegAgent(DQNAgent):
    """
    L2 regularisation baseline.
    Adds an L2 penalty on weights to prevent weight-norm explosion,
    a known precursor to plasticity loss.
    """

    def __init__(self, *args, l2_coeff=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_coeff = l2_coeff
        self.name = "L2-Regularisation"

    @tf.function
    def _update(self, obs, actions, rewards, next_obs, dones):
        next_q = tf.reduce_max(self.target_net(next_obs, training=False), axis=1)
        targets = rewards + self.gamma * next_q * (1.0 - dones)

        with tf.GradientTape() as tape:
            q_vals = self.online_net(obs, training=True)
            idx    = tf.stack(
                [tf.range(tf.shape(actions)[0]), actions], axis=1
            )
            q_pred = tf.gather_nd(q_vals, idx)
            td_loss = tf.reduce_mean(tf.square(targets - q_pred))
            # L2 regularisation term
            l2_loss = tf.add_n([
                tf.reduce_sum(tf.square(w))
                for w in self.online_net.trainable_variables
                if len(w.shape) >= 2
            ])
            loss = td_loss + self.l2_coeff * l2_loss

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self._apply_gradients(grads)
        return td_loss  # return TD loss for fair comparison
