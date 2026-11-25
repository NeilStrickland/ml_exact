import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import plotly.graph_objects as go

class PL1Model(keras.Model):
    """
    An instance of this class is a Keras model that implements a 
    piecewise linear function of one variable. The function is defined 
    by the parameters a, m, and c, where:
    - a is a 1D array of breakpoints (the x-coordinates where the slope 
      changes).  This is assumed to be sorted in ascending order.
    - m is a 1D array of slopes: m[i] is the slope of the function in 
      the interval [a[i], a[i+1]), or the [a[i], infinity) if 
      i is the last index.  
    - c is a scalar representing the value of the function on
      the interval (-infinity, a[0]].
    """
    def __init__(self, a, m, c, **kwargs):
        self.a = a
        self.m = m
        self.c = c
        self.r = int(self.m.shape[0])
        mid_weights = np.ones((1,self.r))
        mid_bias = -self.a
        out_weights = np.diff(m, prepend=0).reshape(self.r,1)
        out_bias = np.array([c])

        inp = keras.Input(shape=(1,))
        mid = keras.layers.Dense(
            units=self.r, 
            activation='relu',
            kernel_initializer=keras.initializers.Constant(mid_weights), # type: ignore
            bias_initializer=keras.initializers.Constant(mid_bias) # type: ignore
            )(inp)
        out = keras.layers.Dense(
            units=1, 
            activation='linear',
            kernel_initializer=keras.initializers.Constant(out_weights), # type: ignore
            bias_initializer=keras.initializers.Constant(out_bias) # type: ignore
            )(mid)
        super().__init__(inputs=inp, outputs=out, **kwargs)

    def plot(self, fig=None, xs=None, **kwargs):
        if fig is None:
            fig = go.Figure()
        if xs is None:
            w = self.a[-1] - self.a[0]
            x0 = self.a[0] - 0.1*w
            x1 = self.a[-1] + 0.1*w
            xs = np.linspace(x0, x1, 100)
        ys = self(xs).numpy().reshape(-1)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='PL1Model', **kwargs))
        return fig


class SubsetMin(keras.layers.Layer):
    def __init__(self, subsets, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.subsets = subsets
        self.axis = axis

    def build(self, input_shape):
        self.subsets_tensor = keras.ops.convert_to_tensor(self.subsets, dtype="int32")
        super().build(input_shape)

    def call(self, inputs):
        gathered = keras.ops.take(inputs, self.subsets_tensor, axis=self.axis)
        reduce_axis = self.axis if self.axis >= 0 else inputs.ndim + self.axis
        return keras.ops.min(gathered, axis=reduce_axis + 1)

    def get_config(self):
        config = super().get_config()
        config.update({"subsets": self.subsets, "axis": self.axis})
        return config

class PLModel(keras.Model):
    """
    An instance of this class represents a piecewise linear 
    function from a convex subset X of R^d to R.  This is 
    specified by the following data:
    - a is an array of shape (n, d) whose rows are the 
      vertices of a triangulation of X
    - f is an array of shape (n,) whose entries are the 
      values of the function at the vertices a[i].
    - s is an array of shape (m, d+1).  Each row is the list 
      of indices of the vertices of a maximal simplex of the 
      triangulation.
    """
    def __init__(self, a, f, s, **kwargs):
        self.a = np.array(a)
        self.f = np.array(f)
        self.s = np.array(s)
        self.n, self.d = a.shape
        self.m = s.shape[0]
        if not(f.shape == (self.n,) and s.shape == (self.m, self.d + 1)):
            raise ValueError("Incompatible array shapes")
        self.aa = np.hstack((self.a, np.ones((self.n, 1))))
        self.b = np.array([np.linalg.inv(self.aa[self.s[i]]).T 
                           for i in range(self.m)])
        self.stars = [[]] * self.n
        for i in range(self.m):
            for j in range(self.d+1):
                self.stars[self.s[i,j]].append(i*(self.d+1)+j)
        ms = max(len(star) for star in self.stars)
        for i in range(self.n):
            if self.stars[i] == []:
                raise ValueError(f"Vertex {i} is not in any simplex")
            self.stars[i] += [self.stars[i][-1]] * (ms - len(self.stars[i]))
        self.weights0 = self.b[:,:,:-1].reshape(self.m * (self.d+1), self.d).T
        self.bias0 = self.b[:,:,-1].reshape(self.m * (self.d+1))

        self.weights1 = np.zeros((self.m, self.d+1, self.n))
        for i in range(self.m):
            for j in range(self.d+1):
                self.weights1[i,j,self.s[i,j]] = 1
        self.weights1 = self.weights1.reshape(self.m * (self.d+1), self.n)

        inp = keras.Input(shape=(self.d,))
        dense = keras.layers.Dense(
            units=self.m * (self.d+1), 
            activation='relu',
            kernel_initializer=keras.initializers.Constant(self.weights0), # type: ignore
            bias_initializer=keras.initializers.Constant(self.bias0), # type: ignore
            name="dense",
        )(inp)

        smin = SubsetMin(
            subsets=self.stars,
            name="smin"
        )(dense)

        out = keras.layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer=keras.initializers.Constant(self.f.reshape(self.n, 1)), # type: ignore
            use_bias=False,
            name="out"
        )(smin)

        super().__init__(inputs=inp, outputs=out, **kwargs)
