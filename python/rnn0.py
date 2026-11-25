import tensorflow as tf
import numpy as np

# This layer computes partial sums.  In more detail, if 
# u is a tensorised version of [[[a0],[a1],[a2],..]]
# then L(u) will be a tensorised version of 
# [[[s0],[s1],[s2],...]], where s0=a0, s1=a0+a1, s2=a0+a1+a2, ...

L = tf.keras.layers.SimpleRNN(1,activation='linear', return_sequences=True)
L.build((None, None, 1))
L.set_weights([np.array([[1]]), np.array([[1]]), np.array([0])])

n = 5
u0 = [2 ** i for i in range(n)]
print(u0)
u = tf.reshape(tf.convert_to_tensor(u0, dtype=tf.float32), [1, n, 1])
v = L(u)
v0 = tf.reshape(v, [-1]).numpy().tolist()
print(v0)

# This layer also computes partial sums, except that the partial sums
# get reset to zero whenever a nonpositive value is encountered.
# The weight matrices are p = [0,1,1] and q = [0,0,1].  The 
# interpretation is as follows.  We have accumulated a partial sum h0
# and we see a new entry x.  We define 
# [z,r,h] = x * p + h0 * q = [0,x,x + h0]
# We then apply the clipping function to r (and also to z, but that 
# is unused).


def clip0(x):
    return min(1, max(0, x))


def clip(x):
    return tf.reshape(tf.map_fn(clip0, tf.reshape(x, [-1])), x.shape)


G = tf.keras.layers.GRU(1, activation='relu', recurrent_activation=clip, return_sequences=True)
G.build((None, None, 1))
G.set_weights([np.array([[0, 1, 1]]), np.array([[0, 0, 1]]), np.array([[0, 0, 0], [0, 0, 0]])])
u0 = [1, 10, 100, 0, 1, 2, 4, -1, -2, 11111, 1110, 100]
print(u0)
u = tf.reshape(tf.convert_to_tensor(u0,dtype=tf.float32), [1, -1, 1])
v = G(u)
v0 = tf.reshape(v, [-1]).numpy().tolist()
print(v0)