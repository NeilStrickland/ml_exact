import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from keras import layers, ops, initializers
import itertools
from plotly.subplots import make_subplots
from plotly import graph_objects as go

class Letter:
    """
    The letters iyjcolhtux can be represented in a reasonable way 
    as 3x3 pixel images with entries in {0,1}.  This class represents 
    3x3 array of image, which might or might not be one of those
    letters.  If it is one of those letters then the index attribute 
    will be an integer from 0 to 9, and the value attribute will be 
    the corresponding letter.  If it is not one of those letters then 
    the index and value attributes will be None.  The image attribute
    is always a 3x3 array of 0s and 1s.
    """
    values = "?iyjcolhtux"
    num_values = len(values)
    size = 3
    area = size*size
    images = np.array([
        [[0,0,0],[0,0,0],[0,0,0]],
        [[0,1,0],[0,1,0],[0,1,0]],
        [[1,0,1],[0,1,0],[0,1,0]],
        [[0,0,1],[0,0,1],[1,1,1]],
        [[1,1,1],[1,0,0],[1,1,1]],
        [[1,1,1],[1,0,1],[1,1,1]],
        [[1,0,0],[1,0,0],[1,1,1]],
        [[1,0,1],[1,1,1],[1,0,1]],
        [[1,1,1],[0,1,0],[0,1,0]],
        [[1,0,1],[1,0,1],[1,1,1]],
        [[1,0,1],[0,1,0],[1,0,1]]
        ], dtype=np.int32)

    lookup_table = dict(zip(
        map(lambda x: tuple(int(j) for j in x.reshape(-1)), images[1:]), 
        range(1, num_values)))

    def __init__(self, index=None, value=None, image=None):
        if index is not None:
            if index < 0 or index >= len(Letter.values):
                raise ValueError(f"Letter index {index} out of range")
            self.index = index
            self.value = Letter.values[index]
            self.image = Letter.images[index]
        elif value is not None:
            # set i to the position of value in values, or -1 if not found
            i = Letter.values.find(value)
            if i == -1:
                raise ValueError(f"Letter {value} not recognised")
            self.index = i
            self.value = value
            self.image = Letter.images[i]
        elif image is not None:
            self.image = np.array(image)
            i = tuple(int(j) for j in self.image.reshape(-1))
            if i in Letter.lookup_table:
                self.index = Letter.lookup_table[i]
                self.value = Letter.values[self.index]
            else:
                self.index = 0
                self.value = '?'
        else:
            self.index = 0
            self.value = '?'
            self.image = Letter.images[0]

    def __str__(self):
        return self.value

    def __repr__(self):
        if self.index > 0:
            return f"Letter({self.value})"
        else:
            i = tuple(int(j) for j in self.image.reshape(-1))
            i = [i[0:3], i[3:6], i[6:9]]
            return f"Letter(image={i})"

    def one_hot(self):
        """
        Return a one-hot vector of length Letter.num_values, with a 1 
        in the position corresponding to the letter's index.
        """
        v = np.zeros(Letter.num_values)
        v[self.index] = 1
        return v

    def show(self, fig=None, x0=0, y0=0):
        if fig is None:
            fig = go.Figure()
            fig.update_layout(width=300, height=300, margin=dict(l=2,r=2,b=2,t=2))
            fig.update_yaxes(scaleanchor="x",scaleratio=1)
            fig.update_xaxes(range=[0,Letter.size], showticklabels=False, showgrid=True, zeroline=False)
            fig.update_yaxes(range=[0,Letter.size], showticklabels=False, showgrid=True, zeroline=False, autorange="reversed")
        p = Letter.size
        xs = list(x0 + i for i in range(p+1))
        ys = list(y0 + i for i in range(p+1))
        im = go.Heatmap(
            x=xs, y=ys, z=self.image, 
            colorscale='Greys', showscale=False
        )
        fig.add_trace(im)
        return fig

    @staticmethod
    def list_good():
        return [Letter(index=i) for i in range(1, Letter.num_values)]

    @staticmethod
    def list_all():
        all_images = np.array(list(itertools.product(*[[0,1]]*9))).reshape(512,3,3)
        return [Letter(image=i) for i in all_images]

    @staticmethod
    def dataset(good=False):
        if good:
            letters = Letter.list_good()
        else:
            letters = Letter.list_all()
        x = np.array([l.image for l in letters])
        y = np.array([l.one_hot() for l in letters])
        return x, y

class Text:
    """
    A text is an array of values in {0,1}, of shape (h, w) say.
    The index (i, j) is to be thought of as the pixel in row i 
    and column j.  Typically h will be between 3 and 10, and w
    will be larger.  The array is expected to contain 3x3 blocks
    representing letters as in the Letter class, arranged in a
    rough line but possibly with some noise. 
    """

    def __init__(self, image=None, letters=None, ragged=False, height=None, width=None):
        if image is not None:
            self.image = np.array(image)
            self.width, self.height = self.image.shape
        elif letters is not None:
            if ragged:
                self.height = 2 * Letter.size
                self.width = (Letter.size + 2) * len(letters)
                if height is not None and height > self.height:
                    self.height = height
                if width is not None and width > self.width:
                    self.width = width
                self.image = np.zeros((self.height, self.width), dtype=np.int32)
                x = np.random.randint(0, 2)
                for i, letter in enumerate(letters):
                    y = np.random.randint(0, self.height-Letter.size)
                    self.add_letter(letter, y, x)
                    x += Letter.size + np.random.randint(1, 3)
                    y = np.random.randint(0, Letter.size + 1)
            else:
                self.height = Letter.size
                self.width = (Letter.size + 1) * len(letters)
                if height is not None and height > self.height:
                    self.height = height
                if width is not None and width > self.width:
                    self.width = width
                self.image = np.zeros((self.height, self.width), dtype=np.int32)
                for i, letter in enumerate(letters):
                    self.add_letter(letter, 0, (Letter.size + 1)*i)
        else:
            self.width = width
            self.height = height
            self.image = np.zeros((self.height, self.width), dtype=np.int32)

    def add_letter(self, letter, i, j):
        """
        Add a letter to the text at position (i, j).  The letter is
        a Letter object.  The position (i, j) is the top left corner
        of the 3x3 block representing the letter.
        """
        if isinstance(letter, str):
            letter = Letter(value=letter)
        elif isinstance(letter, int):
            letter = Letter(index=letter)
        self.image[i:i+Letter.size, j:j+Letter.size] = letter.image

    def show(self, fig=None):
        d = np.round(800 / max(self.width, self.height))
        if fig is None:
            fig = go.Figure()
            fig.update_layout(width=self.width * d, height=self.height * d, margin=dict(l=2,r=2,b=2,t=2))
            fig.update_yaxes(scaleanchor="x",scaleratio=1)
            fig.update_xaxes(range=[0,self.width], showticklabels=False, showgrid=True, zeroline=False)
            fig.update_yaxes(range=[0,self.height], showticklabels=False, showgrid=True, zeroline=False, autorange="reversed")
        xs = list(range(self.width+1))
        ys = list(range(self.height+1))
        im = go.Heatmap(x=xs, y=ys,z=self.image, 
                        colorscale='Greys', showscale=False)
        fig.add_trace(im)
        return fig


class LetterModel(keras.Model):
    """
    An exact model that recognises letters from 3x3 images.

    The input is a 3x3 image with entries in {0,1}.  The output is a
    vector of length 11, where the first 10 entries are the probabilities
    that the image is a particular letter from the string Letter.values.
    The last entry is the probability that the image is not a letter.  
    These probabilities will always be close to 0 or 1,
    so we can use argmax to choose the letter.
    """

    def __init__(self, **kwargs):
        mid_weights = (2 * Letter.images - 1).reshape(-1, Letter.area).T.astype(np.float32)
        mid_bias = (1 - Letter.images.sum(axis=(1,2))).astype(np.float32)
        mid_bias[0] = 0
        out_weights = np.eye(Letter.num_values, dtype=np.float32)
        out_weights[1:,0] = -1
        out_bias = np.zeros(Letter.num_values, dtype=np.float32)
        out_bias[0] = 1
        super().__init__(**kwargs)
        self.inp = keras.Input(shape=(Letter.size, Letter.size))
        self.flatten = keras.layers.Flatten()
        self.mid = keras.layers.Dense(
            units=Letter.num_values, 
            activation='relu',
            kernel_initializer=initializers.Constant(mid_weights), 
            bias_initializer=initializers.Constant(mid_bias),
            trainable=False
        )
        self.out = keras.layers.Dense(
            units=Letter.num_values, 
            activation='relu',
            kernel_initializer=initializers.Constant(out_weights), 
            bias_initializer=initializers.Constant(out_bias),
            trainable=False
        )
        self(np.zeros((1, Letter.size, Letter.size))) # call the model once to build it

    def call(self, x, training=False):
        y = self.flatten(x)
        y = self.mid(y, training=training)
        y = self.out(y, training=training)
        return y

    def classify(self, x):
        y = self(x.reshape((1, Letter.size, Letter.size)), training=False)
        i = np.argmax(y, axis=1)
        return Letter.values[i[0]]
        
    def check(self):
        x, y = Letter.dataset()
        y_pred = self(x, training=False)
        return np.array_equal(y_pred, y)
    
class TextCountModel(keras.Model):
    """
    An exact model that counts letters in a text of the specified size.
    The output is a vector of length 11, where the first 10 entries are the
    numbers of instances of the letters in the string letters defined above,
    and the last entry is the number of 3x3 blocks that do not contain a letter.
    """

    def __init__(self, height=6, width=15, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        LM = LetterModel()
        self.mid_weights, self.mid_bias = LM.mid.get_weights()
        self.mid_weights = self.mid_weights.reshape(Letter.size, Letter.size, 1, Letter.num_values)
        self.hot_weights, self.hot_bias = LM.out.get_weights()
        self.inp = keras.Input(shape=(self.height, self.width))
        self.reshape = keras.layers.Reshape((self.height, self.width, 1))
        self.mid = keras.layers.Conv2D(
            filters=Letter.num_values, 
            kernel_size=[Letter.size, Letter.size], 
            strides=[1, 1], 
            padding='valid', 
            activation='relu',
            kernel_initializer=initializers.Constant(self.mid_weights),
            bias_initializer=initializers.Constant(self.mid_bias)
        )
        self.hot = keras.layers.Dense(
            units=Letter.num_values, 
            activation='relu',
            kernel_initializer=initializers.Constant(self.hot_weights),
            bias_initializer=initializers.Constant(self.hot_bias)
        )
        self.out = layers.Lambda(lambda x: ops.sum(x, axis=(0, 1, 2)))

    def call(self, x, training=False):
        y = np.array(x).astype(np.float32)
        y = self.reshape(y)
        y = self.mid(y, training=training)
        y = self.hot(y, training=training)
        y = self.out(y)
        return y

    def read(self, text):
        x = np.zeros((1, self.height, self.width), dtype=np.float32)
        x[0, :text.height, :text.width] = text.image
        y = self(x=x, training=False)
        n = Letter.num_values
        s = ""
        for i in tuple(range(1, n)) + (0,):
            if y[i] > 0:
                if len(s) > 0:
                    s += "; "
                s += f"{Letter.values[i]}: {int(y[i])}"
        return s

class ZeroFilterA(keras.Model):
    """
    A model that returns the last p nonzero values in a sequence x,
    which is expected to consist of nonnegative integers.

    The layer nz converts x to a matrix with two rows. The first row is x and the 
    second row is a=I(x=0)=(1 - x)_+

    The layer gru maintains a hidden vector h of length p, which is initially zero.
    It gets updated for every entry (x, a) in the input.  The candidate update is
    hh = Lh + x*e, where L is the left shift operator and e is the last basis vector.
    The update is h = (1-a)*hh + a*h.  

    In terms of the standard notation for a GRU, we have:
    z = a * np.ones(p)
    r = np.ones(p)
    hh = Lh + x*e

    (There is some ambiguity in the literature about z vs 1-z.  Keras, Pytorch
    and the original paper by Cho et al. have h = (1-z)*hh + z*h, i.e. we 
    update with probability 1-z.  However, Wikipedia has h = z*hh + (1-z)*h.)
    
    The weights can be explained as follows:
    v2 = [v2z | v2r | v2h] = [[0 | 0 | e], [np.ones(p) | 0 | 0]]
    w2 = [w2z | w2r | w2h] = [0 | 0 | L]
    b2 = [b2z | b2r | b2h] = [0 | np.ones(p) | 0]
    z  = [x,a] @ v2z + h @ w2z + b2z
    r  = [x,a] @ v2r + h @ w2r + b2r
    hh = [x,a] @ v2h + (r * h) @ w2h + b2h
    """

    def __init__(self, p, **kwargs):
        self.p = p
        super().__init__(**kwargs)
        self.inp = keras.Input(shape=(None,1))
        self.nz = keras.layers.Dense(
            units=2, 
            activation='relu',
            kernel_initializer=initializers.Constant(np.array([[1.,-1.]])), 
            bias_initializer=initializers.Constant(np.array([0.,1.]))
        )
        v2 = np.zeros((2,3*p))
        v2[0,3*p-1] = 1
        v2[1,0:p] = np.ones(p)
        w2 = np.zeros((p,3*p))
        w2[1:p,2*p:3*p-1] = np.eye(p-1)
        b2 = np.zeros(3*p)
        b2[p:2*p] = np.ones(p)
        self.gru = keras.layers.GRU(
            units=p, 
            activation='linear', 
            recurrent_activation='linear', 
            reset_after=False,
            kernel_initializer=initializers.Constant(v2),
            recurrent_initializer=initializers.Constant(w2),
            bias_initializer=initializers.Constant(b2)
        )

    def call(self, x, training=False):
        y = self.nz(x, training=training)
        y = self.gru(y, training=training)
        return y

class ZeroFilterB(keras.Model):
    """
    A model that takes a p-vector x and moves any zeros to the end.

    The layer nz converts x to a matrix with two rows. The first row is x and the 
    second row is a=I(x=0)=(1 - x)_+

    The layer gru maintains a hidden vector h of length 3*p, which we group
    into three p-vectors [l0 | l1 | m].  In fact l0 and l1 will be equal.
    Initially, we have m = 0 and l = l0 = l1 is the first basis vector.
    Later on, m will contain all the nonzero xs that we have seen so far, 
    and l will be a basis vector in the place where the next nonzero
    x should go.

    The vector h gets updated for every entry (x, a) in the input.  The candidate
    update is hh = [Rl | Rl | m+x*l], where R is the right shift operator.
    The update is h = (1-a)*hh + a*h.  

    In terms of the standard notation for a GRU, we have:
    z = a * np.ones(3*p)
    r = [np.ones(p) | x * np.ones(p) | np.ones(p)]
    r * h = [l | x * l | m]
    hh = [Rl | Rl | m+x*l] 
    """
    def __init__(self, p, **kwargs):
        self.p = p
        super().__init__(**kwargs)
        self.inp = keras.Input(shape=(p,1))
        self.nz = keras.layers.Dense(
            units=2, 
            activation='relu',
            kernel_initializer=initializers.Constant(np.array([[1.,-1.]])),
            bias_initializer=initializers.Constant(np.array([0.,1.]))
        )
        h0 = np.zeros((1,3*p))
        h0[0,0] = 1
        h0[0,p] = 1
        self.h0 = h0
        v2 = np.zeros((2,9*p))
        v2[0,4*p:5*p] = np.ones(p)
        v2[1,0  :3*p] = np.ones(3*p)
        w2 = np.zeros((3*p,9*p))
        w2[0  :p-1, 6*p+1:7*p] = np.eye(p-1)
        w2[0  :p-1, 7*p+1:8*p] = np.eye(p-1)
        w2[p  :2*p, 8*p  :9*p] = np.eye(p)
        w2[2*p:3*p, 8*p  :9*p] = np.eye(p)
        b2 = np.zeros(9*p)
        b2[3*p:4*p] = np.ones(p)
        b2[5*p:6*p] = np.ones(p)
        self.gru = keras.layers.GRU(
            units=3*p, 
            activation='linear', 
            recurrent_activation='linear', 
            reset_after=False,
            kernel_initializer=initializers.Constant(v2), 
            recurrent_initializer=initializers.Constant(w2), 
            bias_initializer=initializers.Constant(b2)
        )
        self.compile(loss='mean_squared_error', metrics=['accuracy'])

    def call(self, x, training=False):
        y = self.nz(x, training=training)
        y = self.gru(y, initial_state=self.h0, training=training)
        y = y.numpy().reshape(-1)[-self.p:]
        return y

class TextModel(keras.Model):
    def __init__(self, height, width, **kwargs):
        self.height = height
        self.width = width

        LM = LetterModel()
        self.mid_weights, self.mid_bias = LM.mid.get_weights()
        self.mid_weights = self.mid_weights.reshape(Letter.size, Letter.size, 1, Letter.num_values)
        self.hot_weights, self.hot_bias = LM.out.get_weights()
        inp = keras.Input(shape=(self.height, self.width))
        reshape0 = keras.layers.Reshape((self.height, self.width, 1))(inp)

        # In the descriptions below:
        # h = self.height, w = self.width, n = Letter.num_values, s = Letter.size

        # mid
        # output shape : (h-s+1, w-s+1, n)
        # For i > 0, the output in position (x, y, i) is 1 if the sxs block starting 
        # at (x, y) is the letter with index i, and 0 otherwise.  The value at 
        # (x, y, 0) is always zero.    
        mid = keras.layers.Conv2D(
            filters=Letter.num_values, 
            kernel_size=[Letter.size, Letter.size], 
            strides=[1, 1], 
            padding='valid', 
            activation='relu',
            kernel_initializer=initializers.Constant(self.mid_weights),
            bias_initializer=initializers.Constant(self.mid_bias)
        )(reshape0)

        # hot
        # output shape : (h-s+1, w-s+1, n)
        # For i > 0, the output in position (x, y, i) is 1 if the sxs block starting 
        # at (x, y) is the letter with index i, and 0 otherwise.  The value at 
        # (x, y, 0) is 0 for a good letter, otherwise 1.    
        hot = keras.layers.Dense(
            units=Letter.num_values, 
            activation='relu',
            kernel_initializer=initializers.Constant(self.hot_weights),
            bias_initializer=initializers.Constant(self.hot_bias)
        )(mid)

        # val
        # output shape : (h-s+1, w-s+1, 2)
        # If the sxs block starting at (x, y) is the letter of index i > 0, then the 
        # output in positions (x, y, 0) and (x, y, 1) is (i, i-1).  If the 
        # block is not a letter then the output is (0, 0).
        w = np.zeros((2, Letter.num_values), dtype=np.float32)
        w[0] = np.arange(Letter.num_values)
        w[1,1:] = np.arange(Letter.num_values-1)
        val = keras.layers.Dense(
            units = 2,
            activation='linear',
            kernel_initializer=initializers.Constant(w.T)
        )(hot)

        # pooling
        # output shape : (1, w-s+1, 2)
        # If the vertical strip of shape hxs starting at (0, y) contains the 
        # letter of index i > 0, then the output in positions (0, y, 0) and (0, y, 1) 
        # is (i, i-1).  If the strip contains several letters, then the one of 
        # largest index is chosen.  If the strip contains no letters then the
        # output is (0, 0).
        pooling = keras.layers.MaxPooling2D(
            pool_size=(self.height,1), 
            strides=(self.height,1), 
            padding='same',
            name="pooling"
        )(val)

        # nz
        # output shape : (1, w-s+1, 2)
        # If the vertical strip of shape hxs starting at (0, y) contains the 
        # letter of index i > 0, then the output in positions (0, y, 0) and (0, y, 1) 
        # is (i, 0).  If the strip contains several letters, then the one of 
        # largest index is chosen.  If the strip contains no letters then the
        # output is (0, 1).
        nz = keras.layers.Dense(
            units = 2,
            activation='linear',
            kernel_initializer=initializers.Constant(np.array([[1,-1],[0,1]])),
            bias_initializer=initializers.Constant(np.array([0,1])),
            trainable=False
        )(pooling)

        p = self.width - Letter.size + 1
        reshape1 = keras.layers.Reshape((p, 2))(nz)

        h = np.zeros((1,3*p))
        h[0,0] = 1
        h[0,p] = 1

        v = np.zeros((2,9*p))
        v[0,4*p:5*p] = np.ones(p)
        v[1,0  :3*p] = np.ones(3*p)

        w = np.zeros((3*p,9*p))
        w[0  :p-1, 6*p+1:7*p] = np.eye(p-1)
        w[0  :p-1, 7*p+1:8*p] = np.eye(p-1)
        w[p  :2*p, 8*p  :9*p] = np.eye(p)
        w[2*p:3*p, 8*p  :9*p] = np.eye(p)

        b = np.zeros(9*p)
        b[3*p:4*p] = np.ones(p)
        b[5*p:6*p] = np.ones(p)
        gru = keras.layers.GRU(
            units=3*p, 
            activation='linear', 
            recurrent_activation='linear', 
            reset_after=False,
            kernel_initializer=initializers.Constant(v), 
            recurrent_initializer=initializers.Constant(w), 
            bias_initializer=initializers.Constant(b),
            trainable=False
        )(reshape1, initial_state=h)

        chop = keras.layers.Lambda(lambda x: x[:, -p:])(gru)

        super().__init__(inputs=inp, outputs=chop, **kwargs)

    def read(this, text):
        """
        Read a text and return a string.
        """
        x = np.zeros((1, this.height, this.width), dtype=np.float32)
        x[0, :text.height, :text.width] = text.image
        y = this(x, training=False).numpy()[0]
        s = ''
        for i in y:
            if i == 0:
                return s
            s += Letter.values[int(i)]
        return s


