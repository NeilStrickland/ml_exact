# Here we build a handcrafted model that computes all 16
# possible boolean functions of two boolean variables.

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from keras import layers, ops, initializers

@keras.saving.register_keras_serializable()
class BoolModel(keras.Model):
    bool_pairs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    bool_funs = [
        [0, 0, 0, 0, "0"],
        [0, 0, 0, 1, "a & b"],
        [0, 0, 1, 0, "a & -b"],
        [0, 0, 1, 1, "a"],
        [0, 1, 0, 0, "-a & b"],
        [0, 1, 0, 1, "b"],
        [0, 1, 1, 0, "a + b"],
        [0, 1, 1, 1, "a | b"],
        [1, 0, 0, 0, "-a & -b"],
        [1, 0, 0, 1, "- (a + b)"],
        [1, 0, 1, 0, "-b"],
        [1, 0, 1, 1, "a | -b"],
        [1, 1, 0, 0, "-a"],
        [1, 1, 0, 1, "-a | b"],
        [1, 1, 1, 0, "-a | -b"],
        [1, 1, 1, 1, "1"]
    ]

    def __init__(self, exact=True, p=3, r=0.001, **kwargs):
        self.exact = exact
        self.p = p
        self.r = r
        self.x_all = np.array(BoolModel.bool_pairs)
        self.y_all = np.array([u[0:4] for u in BoolModel.bool_funs]).T

        super().__init__(**kwargs)
        if exact:
            dense_weights = np.array([[1, 0], [0, 1], [1, 1]]).T
            dense_bias = np.array([0, 0, -1])
            out_weights = np.array([
                [u[2] - u[0], u[1] - u[0], u[3] - u[2] - u[1] + u[0]] 
                for u in BoolModel.bool_funs
            ]).T
            out_bias = np.array([u[0] for u in BoolModel.bool_funs])
            self.inp = layers.Input(shape=(2,))
            self.dense = layers.Dense(
                units=3, 
                activation='relu',
                kernel_initializer=initializers.Constant(dense_weights), # type: ignore[call-overload]
                bias_initializer=initializers.Constant(dense_bias)  # type: ignore[call-overload]
            )
            self.out = layers.Dense(
                units=16, 
                activation='linear',
                kernel_initializer=initializers.Constant(out_weights), # type: ignore[call-overload]
                bias_initializer=initializers.Constant(out_bias)  # type: ignore[call-overload]
            )
        else:
            self.inp = layers.Input(shape=(2,))
            self.dense = layers.Dense(
                units=p, 
                activation='sigmoid'
            )
            self.out = layers.Dense(
                units=16, 
                activation='sigmoid'
            )

        opt = keras.optimizers.Adam(learning_rate=r)
        self.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])

    def get_config(self):
        config = super().get_config()
        config.update({"exact": self.exact, "p": self.p, "r": self.r})
        return config

    def call(self, x, training=False):
        y = self.dense(x, training=training)
        y = self.out(y, training=training)
        return y

    def check(self):
        y_pred = self(self.x_all, training=False)
        if not self.exact:
            y_pred = np.round(y_pred)
        return np.array_equal(y_pred, self.y_all)

    def train(self, epochs: int = 20, verbose: int = 1):
        self.fit(self.x_all, self.y_all, epochs=epochs, verbose=verbose) # type: ignore[arg-type]

    def model_file(self):
        return f'models/bool_model_{self.p}.keras'
    
    def save(self, filepath=None, overwrite=True, zipped: bool | None = None, **kwargs):
        if self.exact:
            raise ValueError("Exact model should not be saved.")
        if filepath is None:
            filepath = self.model_file()
        super().save(
            filepath=filepath,
            overwrite=overwrite,
            zipped=zipped,
            **kwargs,
        )

    @staticmethod
    def load(p=3):
        model_file = f'models/bool_model_{p}.keras'
        return keras.models.load_model(model_file)

