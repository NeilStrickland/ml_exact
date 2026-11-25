# Here we build a handcrafted model that identifies digits
# when drawn as 6x6 images in a standardised way

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers, ops, initializers
from pathlib import Path
import numpy as np
import numbers
from typing import Literal
from plotly.subplots import make_subplots
from plotly import graph_objects as go

src_dir = Path(__file__).resolve().parent
models_dir = (src_dir / 'models').resolve()

N = 6
BATCH_SIZE = 64
SHUFFLE_SIZE = 100
EPOCHS = 20
VALIDATION_SPLIT = 0.1

class Digit:
    """
    An instance of this class represents a digit image of the type used
    in this simplified MNIST task.  

    Attributes:
        value (int): The digit value (0-9).
        params (list): A list of parameters that define the digit's shape.
        image (np.ndarray): A 6x6 array representing the digit image.

    """
    value = None
    params = None
    image = np.zeros((N, N))
    param_length = [4,3,7,7,6,7,5,4,5,5]

    def __init__(self, value, params):
        """
        Initialize a Digit instance.

        Args:
            value (int): The digit value (0-9).
            params (list): A list of parameters that define the digit's shape.
        Raises:
            ValueError: If the value is not a digit or if the 
            params list is not of the correct length.
        """
        if not (isinstance(value, numbers.Integral) and 0 <= int(value) < 10):
            raise ValueError("value is not a digit")

        if not (len(params) == self.param_length[value]):
            raise ValueError("params is not a list of the correct length")
        
        self.value = value
        self.params = np.array(params)
        self.image = np.zeros((N, N))
        i = self.params

        if value==0:
            if not (0 <= i[0] < i[1] - 1 < N - 1 and
                   0 <= i[2] < i[3] - 1 < N - 1):
                raise ValueError("bad dimensions")
            self.hl(i[0], i[1], i[2])
            self.hl(i[0], i[1], i[3])
            self.vl(i[0], i[2], i[3])
            self.vl(i[1], i[2], i[3])
        elif value==1:
            if not (0 <= i[0] < N and 0 <= i[1] < i[2] < N):
                raise ValueError("bad dimensions")
            self.vl(i[0], i[1], i[2])
        elif value==2:
            if not (0 <= i[0] < i[1] < N and
                    i[0] < i[2] < N and
                    0 <= i[3] < i[2] and
                    0 <= i[4] < i[5] - 1 < i[6] - 2 < N - 2):
                raise ValueError("bad dimensions")
            self.hl(i[3], i[2], i[4])
            self.hl(i[0], i[2], i[5])
            self.hl(i[0], i[1], i[6])
            self.vl(i[2], i[4], i[5])
            self.vl(i[0], i[5], i[6])
        elif value==3:
            if not (0 <= i[0] < i[1] < N and
                    0 <= i[2] < i[1] and
                    0 <= i[3] < i[1] and
                    0 <= i[4] < i[5] - 1 < i[6] - 2 < N - 2):
                raise ValueError("bad dimensions")
            self.hl(i[0], i[1], i[4])
            self.hl(i[2], i[1], i[5])
            self.hl(i[3], i[1], i[6])
            self.vl(i[1], i[4], i[6])
        elif value==4:
            if not (0 <= i[0] < i[1] - 1 < N - 1 and
                    0 <= i[2] < i[3] and
                    0 <= i[4] < i[3] < i[5] < N):
                raise ValueError("bad dimensions")
            self.hl(i[0], i[1], i[3])
            self.vl(i[0], i[2], i[3])
            self.vl(i[1], i[4], i[5])
        elif value==5:
            if not (0 <= i[0] < i[1] < N and
                    0 <= i[2] < i[1] and
                    i[2] < i[3] < N and
                    0 <= i[4] < i[5] - 1 < i[6] - 2 < N - 2):
                raise ValueError("bad dimensions")
            self.hl(i[0], i[1], i[6])
            self.hl(i[2], i[1], i[5])
            self.hl(i[2], i[3], i[4])
            self.vl(i[2], i[4], i[5])
            self.vl(i[1], i[5], i[6])
        elif value==6:
            if not (0 <= i[0] < i[1] - 1 < N - 1 and
                    0 <= i[2] < i[3] - 1 < i[4] - 2 < N - 2):
                raise ValueError("bad dimensions")
            self.hl(i[0], i[1], i[4])
            self.hl(i[0], i[1], i[3])
            self.vl(i[0], i[2], i[4])
            self.vl(i[1], i[3], i[4])
        elif value==7:
            if not (0 <= i[0] < i[1] < N and 0 <= i[2] < i[3] < N):
                raise ValueError("bad dimensions")
            self.hl(i[0], i[1], i[2])
            self.vl(i[1], i[2], i[3])
        elif value==8:
            if not (0 <= i[0] < i[1] - 1 < N - 1 and
                    0 <= i[2] < i[3] - 1 < i[4] - 2 < N - 2):
                raise ValueError("bad dimensions")
            self.hl(i[0], i[1], i[2])
            self.hl(i[0], i[1], i[3])
            self.hl(i[0], i[1], i[4])
            self.vl(i[0], i[2], i[4])
            self.vl(i[1], i[2], i[4])
        elif value==9:
            if not (0 <= i[0] < i[1] - 1 < N - 1 and
                    0 <= i[2] < i[3] - 1 < i[4] - 2 < N - 2):
                raise ValueError("bad dimensions")
            self.hl(i[0], i[1], i[3])
            self.hl(i[0], i[1], i[2])
            self.vl(i[0], i[2], i[3])
            self.vl(i[1], i[2], i[4])
        else:
            raise ValueError("value is not a digit")

    def hl(self, i1, i2, j):
        """
        Add a horizontal line to the image from (i1, j) to (i2, j).
        """
        for p in range(i1, i2 + 1):
            self.image[j, p] = 1

    def vl(self, i, j1, j2):
        """
        Add a vertical line to the image from (i, j1) to (i, j2).
        """
        for q in range(j1, j2 + 1):
            self.image[q, i] = 1

    def one_hot(self):
        """
        Return a one-hot encoding of the digit value.
        """
        y = np.zeros((10,), dtype=np.float32)
        y[self.value] = 1
        return y

    @staticmethod
    def list_all_params(d=None):
        """
        If d is None, return a list of all possible parameter combinations for each digit.
        If d is a digit, return a list of all possible parameter combinations for that digit.
        """
        if d==0:
            return [[i0, i1, i2, i3]
                    for i0 in range(N - 2)
                    for i1 in range(i0 + 2, N)
                    for i2 in range(N - 2)
                    for i3 in range(i2 + 2, N)]
        elif d==1:
            return [[i0, i1, i2]
                    for i0 in range(N)
                    for i1 in range(N - 1)
                    for i2 in range(i1 + 1, N)]
        elif d==2:
            return [[i0, i1, i2, i3, i4, i5, i6]
                    for i0 in range(N - 1)
                    for i1 in range(i0 + 1, N)
                    for i2 in range(i0 + 1, N)
                    for i3 in range(i2)
                    for i4 in range(N - 4)
                    for i5 in range(i4 + 2, N - 2)
                    for i6 in range(i5 + 2, N)]
        elif d==3:
            return [[i0, i1, i2, i3, i4, i5, i6]
                    for i0 in range(N - 1)
                    for i1 in range(i0 + 1, N)
                    for i2 in range(i1)
                    for i3 in range(i1)
                    for i4 in range(N - 4)
                    for i5 in range(i4 + 2, N - 2)
                    for i6 in range(i5 + 2, N)]
        elif d==4:
            return [[i0, i1, i2, i3, i4, i5]
                    for i0 in range(N - 2)
                    for i1 in range(i0 + 2, N)
                    for i2 in range(N - 1)
                    for i3 in range(i2 + 1, N)
                    for i4 in range(i3)
                    for i5 in range(i3 + 1, N)]
        elif d==5:
            return [[i0, i1, i2, i3, i4, i5, i6]
                    for i0 in range(N - 1)
                    for i1 in range(i0 + 1, N)
                    for i2 in range(i1)
                    for i3 in range(i2 + 1, N)
                    for i4 in range(N - 4)
                    for i5 in range(i4 + 2, N - 2)
                    for i6 in range(i5 + 2, N)]
        elif d==6:
            return [[i0, i1, i2, i3, i4]
                    for i0 in range(N - 2)
                    for i1 in range(i0 + 2, N)
                    for i2 in range(N - 4)
                    for i3 in range(i2 + 2, N)
                    for i4 in range(i3 + 2, N)]
        elif d==7:
            return [[i0, i1, i2, i3]
                    for i0 in range(N - 1)
                    for i1 in range(i0 + 1, N)
                    for i2 in range(N - 1)
                    for i3 in range(i2 + 1, N)]
        elif d==8:
            return [[i0, i1, i2, i3, i4]
                    for i0 in range(N - 2)
                    for i1 in range(i0 + 2, N)
                    for i2 in range(N - 4)
                    for i3 in range(i2 + 2, N)
                    for i4 in range(i3 + 2, N)]
        elif d==9:
            return [[i0, i1, i2, i3, i4]
                    for i0 in range(N - 2)
                    for i1 in range(i0 + 2, N)
                    for i2 in range(N - 4)
                    for i3 in range(i2 + 2, N)
                    for i4 in range(i3 + 2, N)]
        elif d is None:
            return [Digit.list_all_params(d) for d in range(10)]
        else:
            raise ValueError("d is not a digit")

    @staticmethod
    def list_all(d=None):
        """
        Return a list of all possible Digit instances for each digit.
        """
        if (isinstance(d, numbers.Integral) and 0 <= int(d) < 10):
            return [Digit(d, params) for params in Digit.list_all_params(d)]
        elif d is None:
            return [Digit.list_all(d) for d in range(10)]
        else:
            raise ValueError("d is not a digit")

    @staticmethod
    def random(d=None):
        """
        Return a random Digit instance for each digit.
        """
        rng = np.random.default_rng()
        if (isinstance(d, numbers.Integral) and 0 <= int(d) < 10):
            params = rng.choice(Digit.list_all_params(d))
            return Digit(d, params)
        elif d is None:
            d = rng.integers(0, 10)
            params = rng.choice(Digit.list_all_params(d))
            return Digit(d, params)
        else:
            raise ValueError("d is not a digit")

    @staticmethod
    def xy_dataset():
        l = [Digit.list_all(d) for d in range(10)]
        x = np.array([digit.image for d in range(10) for digit in l[d]]).astype("float32")
        y = np.array([digit.one_hot() for d in range(10) for digit in l[d]]).astype("float32")
        return x, y

class ExactModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        features_spec = [
                [[ 0, -1,  0], [ 1,  1,  1], [ 0, -1,  0], -2,"HM","horizontal middle","Reds"],
                [[ 0, -1,  0], [-1,  1,  1], [ 0, -1,  0], -1,"HL","horizontal left end","Reds"],
                [[ 0, -1,  0], [ 1,  1, -1], [ 0, -1,  0], -1,"HR","horizontal right end","Reds"],
                [[ 0,  1,  0], [-1,  1, -1], [ 0,  1,  0], -2,"VM","vertical middle","Blues"],
                [[ 0, -1,  0], [-1,  1, -1], [ 0,  1,  0], -1,"VT","vertical top end","Blues"],
                [[ 0,  1,  0], [-1,  1, -1], [ 0, -1,  0], -1,"VB","vertical bottom end","Blues"],
                [[-1, -1,  0], [-1,  1,  1], [ 0,  1, -1], -2,"TL","top left corner","Greens"],
                [[ 0, -1, -1], [ 1,  1, -1], [-1,  1,  0], -2,"TR","top right corner","Greens"],
                [[ 0,  1, -1], [-1,  1,  1], [-1, -1,  0], -2,"BL","bottom left corner","Greens"],
                [[-1,  1,  0], [ 1,  1, -1], [ 0, -1, -1], -2,"BR","bottom right corner","Greens"],
                [[-1,  1, -1], [-1,  1,  1], [-1,  1, -1], -3,"RT","right tee","Purples"],
                [[-1,  1, -1], [ 1,  1, -1], [-1,  1, -1], -3,"LT","left tee","Purples"]
        ]
        features_weights = (np.array([f[0:3] for f in features_spec]).
                            transpose((1, 2, 0)).reshape((3, 3, 1, 12)))
        features_bias = np.array([f[3] for f in features_spec])
        self.feature_codes = [f[4] for f in features_spec]
        self.feature_names = [f[5] for f in features_spec]
        self.feature_colors = [f[6] for f in features_spec]

        counts_weights = np.zeros((N, N, 12, 12))
        for j in range(N):
            for k in range(N):
                counts_weights[j, k, 1, 0] = 1
                counts_weights[j, k, 2, 1] = 1
                for p in range(8):
                    counts_weights[j, k, p + 4, p + 2] = 1
                    counts_weights[j, k, 6, 10] =  j
                    counts_weights[j, k, 7, 10] = -j
                    counts_weights[j, k, 8, 10] =  j
                    counts_weights[j, k, 9, 10] = -j
                    counts_weights[j, k, 6, 11] =  j
                    counts_weights[j, k, 7, 11] = -j
                    counts_weights[j, k, 8, 11] =  j
                    counts_weights[j, k, 9, 11] = -j
        counts_bias = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])

        output_weights = np.array([
            [-1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  0,  0],
            [-1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  0,  0],
            [ 1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1, -1],
            [ 1, -1, -1, -1, -1,  1, -1,  1, -1,  1,  0,  0],
            [-1, -1,  1,  1, -1, -1,  1, -1, -1,  1,  0,  0],
            [ 1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1,  1],
            [-1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  0,  0],
            [ 1, -1, -1,  1, -1,  1, -1, -1, -1, -1,  0,  0],
            [-1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  0,  0],
            [-1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  0,  0]
        ]).transpose()
        output_bias = np.array([-3, -1, -6, -5, -4, -5, -4, -2, -5, -4])

        self.reshape = layers.Reshape((N, N, 1))
        self.features = layers.Conv2D(
            filters=12, 
            kernel_size=[3, 3], 
            strides=[1, 1], 
            padding='same', 
            activation='relu',
            kernel_initializer=initializers.Constant(features_weights), # type: ignore[call-overload]
            bias_initializer=initializers.Constant(features_bias) # type: ignore[call-overload]
        ) 
        self.counts = keras.layers.Conv2D(
            filters=12, 
            kernel_size=[N, N], 
            strides=[1, 1], 
            padding='valid', 
            activation='relu',
            kernel_initializer=initializers.Constant(counts_weights), # type: ignore[call-overload]
            bias_initializer=initializers.Constant(counts_bias) # type: ignore[call-overload]
        )

        self.out = keras.layers.Dense(
            units=10, 
            activation='relu',
            kernel_initializer=initializers.Constant(output_weights), # type: ignore[call-overload]
            bias_initializer=initializers.Constant(output_bias) # type: ignore[call-overload]
        )

    def call(self, x, training=False):
        y = self.reshape(x)
        y = self.features(y, training=training)
        y = self.counts(y, training=training)
        y = self.out(y, training=training)
        return y
    
    def extract_features(self, x, training=False):
        y = self.reshape(x)
        y = self.features(y, training=training)
        return y
    
    def classify(self, img):
        # Direct forward pass (no progress bar or dataset loop overhead)
        p = self(img.reshape(1, N, N), training=False)
        return int(ops.convert_to_numpy(ops.argmax(p[0, 0, 0]))) # type: ignore[arg-type]

    def check(self):
        x, y = Digit.xy_dataset()
        return np.array_equal(self.predict(x).reshape((-1, 10)), y)

    def show_features(self, img):
        feature_positions = [
            (1, 2), (1, 1), (1, 3), 
            (2, 4), (1, 4), (3, 4),
            (4, 1), (4, 2), (4, 3), (4, 4),
            (2, 1), (3, 1)
        ]
#        features_model = keras.Model(inputs=self.input, outputs=self.layers[2].output)
#        f = features_model(img.reshape((1, N, N, 1)))[0].numpy()
        f = self.extract_features(img.reshape((1, N, N)))[0]
        fig = make_subplots(
            rows=4, cols=4,
            specs=[
                [{}, {}, {}, {}],
                [{}, {"rowspan": 2, "colspan": 2}, None, {}],
                [{}, None, None, {}],
                [{}, {}, {}, {}],
            ],
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )
        for i, (row, col) in enumerate(feature_positions):
            fig.add_trace(
                go.Heatmap(z=f[:, :, i], zmin=0, zmax=1, 
                           colorscale=self.feature_colors[i], 
                           name = self.feature_codes[i],
                           showscale=False),
                row=row, col=col
            )
        fig.add_trace(
            go.Heatmap(z=img, colorscale="Greys", showscale=False),
            row=2, col=2,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
        fig.update_layout(width=600, height=600, margin=dict(l=0, r=0, t=0, b=0))
        return fig

@keras.saving.register_keras_serializable()
class ApproxModel(keras.Model):
    def __init__(self, p=3, q=3, r=0.001, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.q = q
        self.r = r
        self.reshape = keras.layers.Reshape((N, N, 1))
        self.features = keras.layers.Conv2D(p, [3, 3], [1, 1], 'same', activation='relu')
        self.counts = keras.layers.Conv2D(q, [N, N], [1, 1], 'valid', activation='relu')
        self.out = keras.layers.Dense(10, activation='softmax')
        opt = keras.optimizers.Adam(learning_rate=r)
        self.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    def call(self, x, training=False):
        y = self.reshape(x)
        y = self.features(y, training=training)
        y = self.counts(y, training=training)
        y = self.out(y, training=training)
        return y
    
    def extract_features(self, x):
        y = self.reshape(x)
        y = self.features(y, training=False)
        return ops.convert_to_numpy(ops.sigmoid(y)[0]) # type: ignore[call-overload]

    def extract_embedding(self, x):
        y = self.reshape(x)
        y = self.features(y, training=False)
        y = self.counts(y, training=False)
        return ops.convert_to_numpy(y)

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p, "q": self.q, "r": self.r})
        return config

    def train(self, epochs: int = EPOCHS, verbose = 1):
        x, y = Digit.xy_dataset()
        self.fit(x, y.reshape(-1,1,1,10), epochs=epochs, verbose=verbose) # type: ignore[arg-type]

    def check(self):
        x, y = Digit.xy_dataset()
        z = np.argmax(y.reshape((-1, 10)), axis=1)
        z_pred = np.argmax(self.predict(x).reshape((-1, 10)), axis=1)
        return float(np.sum(z == z_pred)) / len(z)

    def model_file(self):
        return f'models/digits_model_{self.p}_{self.q}.keras'
    
    def save(self, filepath=None, overwrite=True, zipped: bool | None = None, **kwargs):
        if filepath is None:
            filepath = self.model_file()
        super().save(
            filepath=filepath,
            overwrite=overwrite,
            zipped=zipped,
            **kwargs,
        )

    @staticmethod
    def load(p=3, q=3):
        model_file = f'models/digits_model_{p}_{q}.keras'
        return keras.models.load_model(model_file)

    def show_kernels(self):
        p = self.p
        lp = list(range(p+1))
        w = self.features.get_weights()[0].reshape(3, 3, p).transpose(2, 0, 1)
        fig = make_subplots(rows=1, cols=p, horizontal_spacing=0.02)
        fig.update_yaxes(scaleanchor="x",scaleratio=1)
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")

        for i in range(p):
            if p == 3:
                colorscale = ["Reds", "Greens", "Blues"][i]
            else:
                colorscale = "Greys"
            fig.add_trace(
                go.Heatmap(x=lp, y=lp, z=w[i], colorscale=colorscale,
                    zmin=np.min(w[i]), zmax=np.max(w[i]), showscale=False,
                ),
                row=1, col=i + 1,
            )
        return fig

    def show_embedding(self):
        cols = ["blue", "orange", "green", "red", "purple",
                "brown", "pink", "gray", "olive", "cyan"]
        fig = go.Figure()
        for d in range(10):
            ims = np.array([x.image for x in Digit.list_all(d)])
            x = self.extract_embedding(ims).reshape(-1, 3) # type: ignore[call-overload]
            fig.add_trace(go.Scatter3d(
                x=x[:, 0], y=x[:, 1], z=x[:, 2],
                mode="markers", marker=dict(size=2, color=cols[d]), 
                name=str(d),
            ))
        fig.update_layout(
            width=800, height=800,
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Feature 3",
            ),
        )
        return fig

    def show_features(self, img):
        ls = self.layers[:3]
        x = img.reshape((1, N, N))
        f : np.ndarray = self.extract_features(x) # type: ignore[call-overload]
#        features_model = tf.keras.Model(inputs=self.input, outputs=self.layers[2].output)
#        f = tf.math.sigmoid(features_model(img.reshape((1, N, N))))[0].numpy()
        if f.shape[-1] in (3, 4):
            fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
            fig.add_trace(
                go.Heatmap(z=img, colorscale="Greys", showscale=False),
                row=1, col=1,
            )
            rgb = np.clip(f * 255, 0, 255).astype(np.uint8)
            fig.add_trace(go.Image(z=rgb), row=1, col=2)
        else:
            fig = make_subplots(rows=1, cols=f.shape[-1] + 1, horizontal_spacing=0.1)
            fig.add_trace(
                go.Heatmap(z=img, colorscale="Greys", showscale=False),
                row=1, col=1,
            )
            for i in range(f.shape[-1]):
                fig.add_trace(
                    go.Heatmap(z=f[:, :, i], colorscale="Greys", showscale=False),
                    row=1, col=i + 2,
                )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
        fig.update_layout(width=800, height=400, margin=dict(l=0, r=0, t=0, b=0))
        return fig

    def show_features_chart(self):
        fig = make_subplots(
            rows=10, cols=10,
            horizontal_spacing=0.01, vertical_spacing=0.01,
        )
        for d in range(10):
            for i in range(5):
                img = Digit.random(d).image
                f : np.ndarray = self.extract_features(img.reshape((1, N, N))) # type: ignore[call-overload]
                fig.add_trace(
                    go.Heatmap(
                        z=img, colorscale="Greys", zmin=0, zmax=1,
                        showscale=False,
                    ),
                    row=d + 1, col=2 * i + 1,
                )
                rgb = np.clip(f * 255, 0, 255).astype(np.uint8)
                fig.add_trace(
                    go.Image(z=rgb),
                    row=d + 1, col=2 * i + 2,
                )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
        fig.update_layout(width=800, height=800, margin=dict(l=0, r=0, t=0, b=0))
        return fig



