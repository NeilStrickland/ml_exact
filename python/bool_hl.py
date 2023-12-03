# Here we build a handcrafted model that computes all 16
# possible boolean functions of two boolean variables.

# exec(open('python/bool.py').read())

import tensorflow as tf
import numpy as np
import itertools
import math
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

# try to do the boolean functions example in n dimensions

n=3
bool_in = list(itertools.product(*[[0,1]]*n))
bool_funs = list(itertools.product(*[[0,1]]*(2**n)))


def make_n_model(n):
    global n_model
    inputs = tf.keras.Input(shape=(n))
    dense1 = tf.keras.layers.Dense(2**n-1, activation='relu')(inputs)
    dense2 = tf.keras.layers.Dense(2**2**n, activation='linear')(dense1)
    n_model = tf.keras.Model(inputs=inputs, outputs=dense2, name='n_model')
    n_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    bool_in = list(itertools.product(*[[0,1]]*n))
    bool_funs = list(itertools.product(*[[0,1]]*(2**n)))

    # First layer creates 2^n-1 variables
    # -1 as the constant term can be added as bias in the second layer
    # use the general idea that xy=relu(x+y-1), xyz=relu(x+y+z-2) etc.
    w0 = np.array(bool_in)[1:]
    b0 = 1-sum(w0.T)
    n_model.layers[1].set_weights([w0.T,b0])

    # second layer builds a vector of weights for each bool function output
    # to output a total of 2^(2^n) functions

    # build the generalisation of u0 + (u2-u0)x + (u1-u0)y + (u3-u2-u1+u0)xy
    # which will have 2^n terms
    w1 = np.zeros((2**2**n,2**n))
    for j in range(2**n):
        print(repr(j)+"out of"+repr(2**n))
        w1[:,j] = w1[:,j] + np.array(bool_funs)[:,j]
        for k in range(j):
            if all(np.array(bool_in)[j] >= np.array(bool_in)[k]):
                w1[:,j] = w1[:,j] - w1[:,k]
    # drop first term u0 from the weight matrix and add it in as a bias
    w1 = w1[:,1:]
    b1 = np.array(bool_funs)[:,0]
    n_model.layers[2].set_weights([w1.T,b1])


make_n_model(4)

#n_model.predict(np.array([0,1,0]).reshape(1,3),verbose=0)
#np.shape(n_model.predict(np.array([0,1,0]).reshape(1,3),verbose=0))

n_model.layers[2].get_weights()[0]

# evaluate against the original functions
def test_n_model(n):
    make_n_model(n=n)

    bool_in = list(itertools.product(*[[0,1]]*n))
    bool_funs = list(itertools.product(*[[0,1]]*(2**n)))

    m = max(1000, 2**2**n-1)

    x_all = np.array(bool_in)
    y_all = np.array(bool_funs).T
    n_model.evaluate(x_all, y_all)

test_n_model(4)

# for n=5, cannot allocate weights of size 31x4294967296 (2^5, 2^2^5)