# Here we build a handcrafted model that computes all 16
# possible boolean functions of two boolean variables.

# exec(open('python/bool.py').read())

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

src_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.realpath(os.path.join(src_dir, '../models'))

exact_model = None
approx_model = None

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

x_all = np.array(bool_pairs)
y_all = np.array([u[0:4] for u in bool_funs]).T
all_dataset = tf.data.Dataset.from_tensor_slices((x_all, y_all))


def make_exact_model():
    global exact_model
    exact_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3, input_dim=2, activation='relu'),
        tf.keras.layers.Dense(16, activation='linear')
    ])

    W0 = np.array([[1, 0], [0, 1], [1, 1]])
    B0 = np.array([0, 0, -1])

    W1 = np.array([
        [u[2] - u[0], u[1] - u[0], u[3] - u[2] - u[1] + u[0]] for u in bool_funs
    ])
    B1 = np.array([u[0] for u in bool_funs])

    exact_model.layers[0].set_weights([W0.T,B0])
    exact_model.layers[1].set_weights([W1.T,B1])

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    exact_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    exact_model.evaluate(x_all, y_all)


def make_approx_model(p=3, r=0.001):
    global approx_model
    approx_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(p, input_dim=2, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='sigmoid')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    approx_model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])


def train_approx_model(epochs):
    approx_model.fit(all_dataset, epochs=epochs)


def save_approx_model():
    l = approx_model.layers
    p = l[1].input_shape[3]
    model_dir = os.path.join(models_dir, 'digits_model_' + repr(p))
    approx_model.save(model_dir)


def load_approx_model(p=3):
    global approx_model
    model_dir = os.path.join(models_dir, 'digits_model_' + repr(p))
    approx_model = tf.keras.models.load_model(model_dir)


