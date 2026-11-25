# Here we build two handcrafted recurrent models that compute partial sums.

import numpy as np
import keras
from keras import initializers, layers, ops

@keras.saving.register_keras_serializable()
class PartialSumModel(keras.Model):
    """Compute cumulative sums for inputs of shape (batch, timesteps, 1)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn = layers.SimpleRNN(
            units=1,
            activation='linear',
            return_sequences=True,
            kernel_initializer='ones',
            recurrent_initializer='ones',
        )

    def build(self, input_shape):
        self.rnn.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        return self.rnn(x, training=training)


@keras.saving.register_keras_serializable()
def clip(x):
    return ops.clip(x, 0, 1)


@keras.saving.register_keras_serializable()
class ResettingPartialSumModel(keras.Model):
    """Compute partial sums with the original GRU reset gate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The update gate is zero, and the reset gate clips the new input x
        # to [0, 1]. The new sum is relu(x + clip(x) * previous_sum).
        # Nonpositive inputs reset the sum, while inputs at least 1 add to it.
        self.gru = layers.GRU(
            units=1,
            activation='relu',
            recurrent_activation=clip, # pyright: ignore[reportArgumentType]
            return_sequences=True,
            kernel_initializer=initializers.Constant([[0, 1, 1]]), # pyright: ignore[reportArgumentType]
            recurrent_initializer=initializers.Constant([[0, 0, 1]]), # pyright: ignore[reportArgumentType]
        )

    def build(self, input_shape):
        self.gru.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        return self.gru(x, training=training)


