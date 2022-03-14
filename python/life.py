# Here we build a handcrafted model that computes the
# recurrence rule for Conway's Game of Life.  We rely on 
# following formulation of the rule: if the value at a 
# cell is u in {0,1}, and the sum of the values at the
# 8 adjacent cells is v, then the new value of u is 1
# if u + 2v lies in {5,6,7}, and 0 otherwise.  We also use
# the fact that (x-4)_+ - (x-5)_+ - (x-7)_+ + (x-8)_+
# is 1 for x in {5,6,7} and 0 for other integer x.

import tensorflow as tf
import numpy as np


def board_string(B):
    return "\n".join(["".join(["#" if x == 1 else "o" for x in R]) for R in B])


def show_board(B):
    print(board_string(B))
    print("======")


n = 6
m1 = [[1] * 4]
m2 = [[2] * 4]
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((n, n, 1), input_shape=(n, n)),
    tf.keras.layers.Conv2D(4, [3, 3], [1, 1], 'same', input_shape=(n, n, 1), activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Reshape((n, n))
])
model.layers[1].set_weights([np.array([[m2, m2, m2], [m2, m1, m2], [m2, m2, m2]]), np.array([-4, -5, -7, -8])])
model.layers[2].set_weights([np.array([[1], [-1], [-1], [1]]), np.array([0])])

model.compile()
model.summary()

B0 = [[1, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 1],
      [0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 1, 1],
      [0, 0, 0, 1, 0, 0]]

show_board(B0)

for i in range(5):
    B0 = model.predict(np.array([B0]))[0]
    show_board(B0)