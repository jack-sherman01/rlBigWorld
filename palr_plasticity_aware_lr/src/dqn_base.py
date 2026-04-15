"""
Base DQN Agent
==============
Shared Q-network architecture and training logic used by all agent variants.
Supports per-layer learning rate scaling for PALR experiments.
"""

import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer
from plasticity_metrics import HIDDEN_LAYER_INDICES


def build_qnet(obs_dim: int, n_actions: int, hidden_sizes=(128, 128)):
    """Build a simple MLP Q-network with ReLU activations."""
    inp = tf.keras.Input(shape=(obs_dim,))
    x = inp
    for h in hidden_sizes:
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    out = tf.keras.layers.Dense(n_actions)(x)
    return tf.keras.Model(inputs=inp, outputs=out)


class DQNAgent:
    """
    Standard DQN with fixed Adam learning rate.

    Args:
        obs_dim: Observation dimensionality.
        n_actions: Number of discrete actions.
        lr: Adam learning rate.
        gamma: Discount factor.
        buffer_size: Replay buffer capacity.
        batch_size: Training batch size.
        target_update_freq: Steps between target network syncs.
        epsilon_start / epsilon_end / epsilon_decay: Epsilon-greedy schedule.
        seed: Random seed.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5_000,
        hidden_sizes=(128, 128),
        seed: int = 42,
    ):
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.obs_dim  = obs_dim
        self.n_actions = n_actions
        self.gamma    = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.hidden_sizes  = hidden_sizes

        self.online_net = build_qnet(obs_dim, n_actions, hidden_sizes)
        self.target_net = build_qnet(obs_dim, n_actions, hidden_sizes)
        self.target_net.set_weights(self.online_net.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.buffer    = ReplayBuffer(buffer_size, obs_dim)

        self.step_count = 0
        self.episode_count = 0
        self.name = "DQN-FixedLR"

    @property
    def epsilon(self):
        frac = min(1.0, self.step_count / self.epsilon_decay)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.online_net(obs[np.newaxis], training=False).numpy()[0]
        return int(np.argmax(q))

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        loss_val = self._update(obs, actions, rewards, next_obs, dones)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.set_weights(self.online_net.get_weights())

        return loss_val

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
            loss   = tf.reduce_mean(tf.square(targets - q_pred))

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self._apply_gradients(grads)
        return loss

    def _apply_gradients(self, grads):
        """Apply gradients -- subclasses can override for custom LR scaling."""
        self.optimizer.apply_gradients(
            zip(grads, self.online_net.trainable_variables)
        )

    def on_episode_end(self, episode_reward: float):
        """Hook called at end of each episode. Override in subclasses."""
        self.episode_count += 1

    def get_hidden_layer_indices(self):
        return HIDDEN_LAYER_INDICES
