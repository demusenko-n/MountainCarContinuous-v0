import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomUniform

class Actor:
    """Actor (Policy) Model"""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize and build actor model"""
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        self.action_range = self.action_high - self.action_low

        # Build actor model
        self.build_model()

        # Define optimizer
        self.optimizer = optimizers.Adam(learning_rate=0.0001)

    def build_model(self):
        """Build an actor policy network that maps states to actions."""
        # Define input layer
        states = layers.Input(shape=(self.state_size,), name='states')

        # Hidden layers
        net = layers.Dense(units=400, kernel_regularizer=l2(1e-6))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(units=300, kernel_regularizer=l2(1e-6))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Output layer with sigmoid activation
        raw_actions = layers.Dense(
            units=self.action_size,
            activation='sigmoid',
            name='raw_actions',
            kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)
        )(net)

        # Scale output to action space range
        actions = layers.Lambda(
            lambda x: (x * self.action_range) + self.action_low,
            name='actions'
        )(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

    def train(self, states, action_gradients):
        """Update policy parameters using given states and action gradients."""
        with tf.GradientTape() as tape:
            # Forward pass
            actions = self.model(states, training=True)
            # Compute the loss
            loss = -tf.reduce_mean(action_gradients * actions)

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
