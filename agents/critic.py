import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, initializers, backend as K

class Critic:
    """Critic model Q(s,a)"""

    def __init__(self, state_size, action_size):
        """Initialize the Critic model.

        Params:
        =======
            state_size (int): Dimension of observation space
            action_size (int): Dimension of action space
        """
        self.state_size = state_size
        self.action_size = action_size

        # Build the critic network
        self.build_model()

        # Define optimizer
        self.optimizer = optimizers.Adam(learning_rate=0.001)

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layers for state pathway
        net_states = layers.Dense(units=400, kernel_regularizer=regularizers.l2(1e-6))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)
        net_states = layers.Dense(units=300, kernel_regularizer=regularizers.l2(1e-6))(net_states)

        # Add hidden layers for action pathway
        net_actions = layers.Dense(units=400, kernel_regularizer=regularizers.l2(1e-6))(actions)
        net_actions = layers.Dense(units=300, kernel_regularizer=regularizers.l2(1e-6))(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(
            units=1,
            name='q_values',
            kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer=initializers.RandomUniform(minval=-0.003, maxval=0.003)
        )(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

    def train(self, states, actions, target_Q_values):
        """Train the critic network.

        Params:
        =======
            states: Batch of states from the environment
            actions: Batch of actions taken
            target_Q_values: Target Q-values computed from the target networks
        """
        with tf.GradientTape() as tape:
            # Forward pass
            Q_values = self.model([states, actions], training=True)
            # Compute the loss
            loss = tf.reduce_mean(tf.square(target_Q_values - Q_values))

        # Compute gradients
        critic_grad = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(critic_grad, self.model.trainable_variables))

    def get_action_gradients(self, states, actions):
        """Compute the gradients of the Q-values with respect to the actions.

        Params:
        =======
            states: Batch of states
            actions: Batch of actions

        Returns:
        ========
            Gradients of the Q-values with respect to the actions
        """
        with tf.GradientTape() as tape:
            tape.watch(actions)
            Q_values = self.model([states, actions], training=False)
        # Compute the gradient of Q_values with respect to actions
        action_gradients = tape.gradient(Q_values, actions)
        return action_gradients
