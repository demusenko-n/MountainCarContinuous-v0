import numpy as np
import tensorflow as tf
from agents.actor import Actor
from agents.critic import Critic
from ReplayBuffer import ReplayBuffer
from agents.ounoise import OUNoise

class DDPG:
    """Reinforcement learning agent who learns using DDPG"""

    def __init__(self, task):
        """Initialize models"""
        self.env = task
        self.state_size = task.observation_space.shape[0]
        self.action_size = task.action_space.shape[0]
        self.action_high = task.action_space.high
        self.action_low = task.action_space.low

        # Initialize Actor (policy) models
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Initialize Critic (value) models
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay buffer
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

    def reset_episode(self, task):
        """Return state after resetting task"""
        self.noise.reset()
        state_info = task.reset()
        # Handle tuple or dict
        if isinstance(state_info, tuple):
            state = state_info[0]
        elif isinstance(state_info, dict):
            state = state_info['observation']
        else:
            state = state_info
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        # Add experience to memory
        self.memory.add_experience(self.last_state, action, reward, next_state, done)

        # Learn if memory is larger than batch size
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over state
        self.last_state = next_state

    def act(self, state):
        """Returns action using the policy network """
        # Handle state if it's a tuple or dict
        if isinstance(state, tuple):
            state = state[0]
        elif isinstance(state, dict):
            state = state['observation']

        # Ensure state is a NumPy array
        state = np.array(state)

        # Reshape state
        state = np.reshape(state, [-1, self.state_size])

        # Predict action
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())

    def learn(self, experiences):
        # Convert experience tuples to separate arrays for each element
        states = tf.convert_to_tensor(
            np.vstack([e.state for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.state_size)
        )
        actions = tf.convert_to_tensor(
            np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        )
        next_states = tf.convert_to_tensor(
            np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.state_size)
        )
        rewards = tf.convert_to_tensor(
            np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        )
        dones = tf.convert_to_tensor(
            np.array([e.done for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        )

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model(next_states)
        Q_targets_next = self.critic_target.model([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

        # Train critic model
        self.critic_local.train(states, actions, Q_targets)

        # Train actor model (local)
        # Get predicted actions from actor local model
        actions_pred = self.actor_local.model(states)

        # Get action gradients (dQ/da) from the critic
        action_gradients = self.critic_local.get_action_gradients(states, actions_pred)

        # Negate the gradients (since we want to maximize the expected return)
        action_gradients = -action_gradients

        # Train the actor
        self.actor_local.train(states, action_gradients)

        # Soft-update target models
        self.soft_update(self.actor_local.model, self.actor_target.model)
        self.soft_update(self.critic_local.model, self.critic_target.model)

    def soft_update(self, local_model, target_model):
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()
        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = []
        for local_weight, target_weight in zip(local_weights, target_weights):
            new_weight = self.tau * local_weight + (1 - self.tau) * target_weight
            new_weights.append(new_weight)
        target_model.set_weights(new_weights)

    def save_model(self, path):
        self.actor_local.model.save_weights(path)

    def load_model(self, path):
        self.actor_local.model.load_weights(path)

    def act_only(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action)
