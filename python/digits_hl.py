# Here we build a handcrafted model that identifies digits
# when drawn as 6x6 images in a standardised way

# exec(open('python/digits.py').read())
import matplotlib.colors
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
from itertools import combinations

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

src_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.realpath(os.path.join(src_dir,'../models'))

exact_model = None
approx_model = None
approx_model_b = None

N = 6
rng = np.random.default_rng()
BATCH_SIZE = 64
SHUFFLE_SIZE = 100
EPOCHS = 20
VALIDATION_SPLIT = 0.1


def show_digit(A):
    plt.imshow(A, cmap='Greys')
    plt.axis("off")


# horizontal line in position j, from i1 to i2
def hl(A, i1, i2, j):
    for p in range(i1, i2+1):
        A[j, p] = 1


# vertical line in position i, from j1 to j2
def vl(A, i, j1, j2):
    for q in range(j1, j2+1):
        A[q, i] = 1


# one-hot representation of digit d
def ee(d):
    u = np.zeros(10)
    u[d] = 1
    return u


# functions for drawing digits
def f0(i):
    if not(isinstance(i, list) and len(i) == 4):
        raise ValueError("i is not a list of length 4")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] - 1 < N - 1):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[2])
    hl(A, i[0], i[1], i[3])
    vl(A, i[0], i[2], i[3])
    vl(A, i[1], i[2], i[3])
    return A


def f1(i):
    if not(isinstance(i, list) and len(i) == 3):
        raise ValueError("i is not a list of length 3")
    if not(0 <= i[0] < N and
           0 <= i[1] < i[2] < N):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    vl(A, i[0], i[1], i[2])
    return A


def f2(i):
    if not(isinstance(i, list) and len(i) == 7):
        raise ValueError("i is not a list of length 7")
    if not(0 <= i[0] < i[1] < N and
           i[0] < i[2] < N and
           0 <= i[3] < i[2] and
           0 <= i[4] < i[5] - 1 < i[6] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[3], i[2], i[4])
    hl(A, i[0], i[2], i[5])
    hl(A, i[0], i[1], i[6])
    vl(A, i[2], i[4], i[5])
    vl(A, i[0], i[5], i[6])
    return A


def f3(i):
    if not(isinstance(i, list) and len(i) == 7):
        raise ValueError("i is not a list of length 7")
    if not(0 <= i[0] < i[1] < N and
           0 <= i[2] < i[1] and
           0 <= i[3] < i[1] and
           0 <= i[4] < i[5] - 1 < i[6] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[4])
    hl(A, i[2], i[1], i[5])
    hl(A, i[3], i[1], i[6])
    vl(A, i[1], i[4], i[6])
    return A


def f4(i):
    if not(isinstance(i, list) and len(i) == 6):
        raise ValueError("i is not a list of length 6")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] and
           0 <= i[4] < i[3] < i[5] < N ):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[3])
    vl(A, i[0], i[2], i[3])
    vl(A, i[1], i[4], i[5])
    return A


def f5(i):
    if not(isinstance(i, list) and len(i) == 7):
        raise ValueError("i is not a list of length 7")
    if not(0 <= i[0] < i[1] < N and
           0 <= i[2] < i[1] and
           i[2] < i[3] < N and
           0 <= i[4] < i[5] - 1 < i[6] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[6])
    hl(A, i[2], i[1], i[5])
    hl(A, i[2], i[3], i[4])
    vl(A, i[2], i[4], i[5])
    vl(A, i[1], i[5], i[6])
    return A


def f6(i):
    if not(isinstance(i, list) and len(i) == 5):
        raise ValueError("i is not a list of length 5")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] - 1 < i[4] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[4])
    hl(A, i[0], i[1], i[3])
    vl(A, i[0], i[2], i[4])
    vl(A, i[1], i[3], i[4])
    return A


def f7(i):
    if not(isinstance(i, list) and len(i) == 4):
        raise ValueError("i is not a list of length 4")
    if not(0 <= i[0] < i[1] < N and
           0 <= i[2] < i[3] < N):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[2])
    vl(A, i[1], i[2], i[3])
    return A


def f8(i):
    if not(isinstance(i, list) and len(i) == 5):
        raise ValueError("i is not a list of length 5")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] - 1 < i[4] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[2])
    hl(A, i[0], i[1], i[3])
    hl(A, i[0], i[1], i[4])
    vl(A, i[0], i[2], i[4])
    vl(A, i[1], i[2], i[4])
    return A


def f9(i):
    if not(isinstance(i, list) and len(i) == 5):
        raise ValueError("i is not a list of length 5")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] - 1 < i[4] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[3])
    hl(A, i[0], i[1], i[2])
    vl(A, i[0], i[2], i[3])
    vl(A, i[1], i[2], i[4])
    return A


# put functions in a list so can call f[0](i) instead of f0(i)
f = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]

# II generates all the possible input configurations for each digit
# to go into its corresponding f[d]
II = [
    [[i0, i1, i2, i3]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-2)
     for i3 in range(i2+2, N)],
    [[i0, i1, i2]
     for i0 in range(N)
     for i1 in range(N-1)
     for i2 in range(i1+1, N)],
    [[i0, i1, i2, i3, i4, i5, i6]
     for i0 in range(N-1)
     for i1 in range(i0+1, N)
     for i2 in range(i0+1, N)
     for i3 in range(i2)
     for i4 in range(N-4)
     for i5 in range(i4+2, N-2)
     for i6 in range(i5+2, N)],
    [[i0, i1, i2, i3, i4, i5, i6]
     for i0 in range(N-1)
     for i1 in range(i0+1, N)
     for i2 in range(i1)
     for i3 in range(i1)
     for i4 in range(N-4)
     for i5 in range(i4+2, N-2)
     for i6 in range(i5+2, N)],
    [[i0, i1, i2, i3, i4, i5]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-1)
     for i3 in range(i2+1, N)
     for i4 in range(i3)
     for i5 in range(i3+1, N)],
    [[i0, i1, i2, i3, i4, i5, i6]
     for i0 in range(N-1)
     for i1 in range(i0+1, N)
     for i2 in range(i1)
     for i3 in range(i2+1, N)
     for i4 in range(N-4)
     for i5 in range(i4+2, N-2)
     for i6 in range(i5+2, N)],
    [[i0, i1, i2, i3, i4]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-4)
     for i3 in range(i2+2, N)
     for i4 in range(i3+2, N)],
    [[i0, i1, i2, i3]
     for i0 in range(N-1)
     for i1 in range(i0+1, N)
     for i2 in range(N-1)
     for i3 in range(i2+1, N)],
    [[i0, i1, i2, i3, i4]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-4)
     for i3 in range(i2+2, N)
     for i4 in range(i3+2, N)],
    [[i0, i1, i2, i3, i4]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-4)
     for i3 in range(i2+2, N)
     for i4 in range(i3+2, N)]
]


# IM generates the 6x6 grids for each of the inputs in II
IM = [[f[d](i) for i in II[d]] for d in range(10)]
x_all = np.array([img for d in range(10) for img in IM[d]]).astype("float32")
y_all = np.array([[[ee(d)]] for d in range(10) for img in IM[d]]).astype("float32")
all_dataset = tf.data.Dataset.from_tensor_slices((x_all, y_all))
all_dataset = all_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)


def random_image(d=None):
    if d is None:
        d = rng.integers(10)
    i = rng.integers(len(IM[d]))
    return IM[d][i]


def make_exact_model():
    global exact_model
    inputs = tf.keras.Input(shape=(N, N))
    reshape = tf.keras.layers.Reshape((N, N, 1))(inputs)
    features = tf.keras.layers.Conv2D(12, [3, 3], [1, 1], 'same', activation='relu')(reshape)
    counts = tf.keras.layers.Conv2D(12, [N, N], [1, 1], 'valid', activation='relu')(features)
    outputs = tf.keras.layers.Dense(10, activation='relu')(counts)
    exact_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="exact_model")
    
    # first convolution layer
    # these weights are the 12 kernels in exact.pdf (in correct order)
    weights2 = [
            [[ 0, -1,  0], [ 1,  1,  1], [ 0, -1,  0]],
            [[ 0, -1,  0], [-1,  1,  1], [ 0, -1,  0]],
            [[ 0, -1,  0], [ 1,  1, -1], [ 0, -1,  0]],
            [[ 0,  1,  0], [-1,  1, -1], [ 0,  1,  0]],
            [[ 0, -1,  0], [-1,  1, -1], [ 0,  1,  0]],
            [[ 0,  1,  0], [-1,  1, -1], [ 0, -1,  0]],
            [[-1, -1,  0], [-1,  1,  1], [ 0,  1, -1]],
            [[ 0, -1, -1], [ 1,  1, -1], [-1,  1,  0]],
            [[ 0,  1, -1], [-1,  1,  1], [-1, -1,  0]],
            [[-1,  1,  0], [ 1,  1, -1], [ 0, -1, -1]],
            [[-1,  1, -1], [-1,  1,  1], [-1,  1, -1]],
            [[-1,  1, -1], [ 1,  1, -1], [-1,  1, -1]]
    ]
    bias2 = [-2, -1, -1, -2, -1, -1, -2, -2, -2, -2, -3, -3]
    weights2 = np.array(weights2).transpose((1, 2, 0)).reshape((3, 3, 1, 12))
    bias2 = np.array(bias2)
    exact_model.layers[2].set_weights([weights2, bias2])

    # second convolution layer
    # here it is size 12 because it skips 0 and 3 but adds 2 indicator parts,
    # not because there are exactly 12 kernels in the first conv layer
    weights3 = np.zeros((N, N, 12, 12))
    for j in range(N):
        for k in range(N):
            # to sum the counts essentially use an identity matrix of weights
            # however miss out 0 and 3 from counts as they are not needed
            weights3[j, k, 1, 0] = 1
            weights3[j, k, 2, 1] = 1
            for p in range(8):
                weights3[j, k, p + 4, p + 2] = 1

            # edit: move the below assignments in one tab as they do not use p

            # the [,10] and [,11] kernels form an indicator variable (ex 6.1)
            # I(6 and 8 are above 7 and 9)
            weights3[j, k, 6, 10] =  j
            weights3[j, k, 7, 10] = -j
            weights3[j, k, 8, 10] =  j
            weights3[j, k, 9, 10] = -j
            weights3[j, k, 6, 11] =  j
            weights3[j, k, 7, 11] = -j
            weights3[j, k, 8, 11] =  j
            weights3[j, k, 9, 11] = -j
    bias3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])

    exact_model.layers[3].set_weights([weights3, bias3])

    # dense layer
    # evaluates the counts against the expected numbers for each digit
    # 2 and 5 use the indicator parts
    weights4 = np.array([
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
    bias4 = np.array([-3, -1, -6, -5, -4, -5, -4, -2, -5, -4])
    exact_model.layers[4].set_weights([weights4, bias4])

    return exact_model


def exact_classify(img):
    p = exact_model.predict(img.reshape(1, N, N))
    return tf.math.argmax(p[0][0][0]).numpy()


feature_pos = [
    [0.4, 0.8, 0.2, 0.2],
    [0.2, 0.8, 0.2, 0.2],
    [0.6, 0.8, 0.2, 0.2],
    [0.0, 0.4, 0.2, 0.2],
    [0.0, 0.6, 0.2, 0.2],
    [0.0, 0.2, 0.2, 0.2],
    [0.3, 0.2, 0.2, 0.2],
    [0.5, 0.2, 0.2, 0.2],
    [0.3, 0.0, 0.2, 0.2],
    [0.5, 0.0, 0.2, 0.2],
    [0.0, 0.8, 0.2, 0.2],
    [0.8, 0.8, 0.2, 0.2]
]


# creates the greyscale diagram of kernels for the pixel types in exact.pdf
def show_features(img):
    l = exact_model.layers
    f = l[2](l[1](l[0](img.reshape((1, N, N)))))[0].numpy()
    fig = plt.figure(figsize=(8, 8))
    for i in range(12):
        p = feature_pos[i]
        g = 0.02
        p = [p[0]+g, p[1]+g, p[2]-2*g, p[3]-2*g]
        ax = fig.add_axes(p)
        ax.set_axis_off()
        ax.imshow(f[:, :, i], cmap='Greys')
    ax = fig.add_axes([0.3, 0.4, 0.4, 0.4])
    ax.set_axis_off()
    ax.imshow(img, cmap='Greys')


# the perfect model described in exact.pdf
def make_approx_model(p=3, q=3, r=0.001):
    global approx_model
    inputs = tf.keras.Input(shape=(N, N))
    reshape = tf.keras.layers.Reshape((N, N, 1))(inputs)
    features = tf.keras.layers.Conv2D(p, [3, 3], [1, 1], 'same', activation='relu')(reshape)
    # here the counts layer is just functioning like a dense layer
    counts = tf.keras.layers.Conv2D(q, [N, N], [1, 1], 'valid', activation='relu')(features)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(counts)
    approx_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="approx_model")
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    approx_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return approx_model


# edit: switch for verbose
def train_approx_model(v=0):
    approx_model.fit(all_dataset, epochs=EPOCHS, verbose=v)


def check_approx_model():
    z_all = np.argmax(y_all.reshape((-1, 10)), axis=1)
    z_all_pred = np.argmax(approx_model(x_all).numpy().reshape((-1, 10)), axis=1)
    #return np.array_equal(z_all, z_all_pred)
    return sum(z_all != z_all_pred)



def save_approx_model():
    l = approx_model.layers
    p = l[3].input_shape[3]
    q = l[4].input_shape[3]
    model_dir = os.path.join(models_dir, 'digits_model_' + repr(p) + '_' + repr(q))
    approx_model.save(model_dir)


def load_approx_model(p=3, q=3):
    global approx_model
    model_dir = os.path.join(models_dir, 'digits_model_' + repr(p) + '_' + repr(q))
    approx_model = tf.keras.models.load_model(model_dir)
    return approx_model


# shows the trained kernels for the first layer as r, g, b in exact.pdf
# edit: switch for showing the exact values
def show_approx_kernels(mat=0):
    l = approx_model.layers
    p = l[3].input_shape[3]
    w = approx_model.layers[2].get_weights()[0].reshape(3, 3, p).transpose(2, 0, 1)
    fig = plt.figure(figsize=(2*p-1, 1))
    for i in range(p):
        if p == 3:
            cm0 = np.zeros([11, 4])
            for j in range(11):
                cm0[j, i] = j / 10
                cm0[j, 3] = 1
            cm = matplotlib.colors.ListedColormap(cm0)
        else:
            cm = 'Greys'
        x = (2*i - 0.02) / (2*p-1)
        ax = fig.add_axes([x, 0.02, 0.96 / (2*p-1), 0.96])
        ax.set_axis_off()
        ax.imshow(w[i], cmap=cm)
        if mat == 1: print(w[i])


# shows the output for the second layer in 3D in exact.pdf
def show_approx_embedding():
    cols = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
            "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    l = approx_model.layers
    for d in range(10):
        x = l[3](l[2](l[1](l[0](IM[d])))).numpy().reshape(-1, 3)
        ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=cols[d])


def show_approx_features(img):
    l = approx_model.layers
    f = l[2](l[1](l[0](img.reshape((1, N, N)))))[0].numpy()
    p = l[3].input_shape[3]
    fig = plt.figure(figsize=(8, 8))
    for i in range(p):
        x = 0.5 - 0.05 * p + 0.1 * i + 0.002
        ax = fig.add_axes([x, 0.002, 0.096, 0.096])
        ax.set_axis_off()
        ax.imshow(f[:, :, i], cmap='Greys')
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.8])
    ax.set_axis_off()
    ax.imshow(img, cmap='Greys')


def show_approx_features_alt(img):
    l = approx_model.layers
    f = tf.math.sigmoid(l[2](l[1](l[0](img.reshape((1, N, N))))))[0].numpy()
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_axes([0.01, 0.02, 0.48, 0.96])
    ax1.set_axis_off()
    ax1.imshow(img, cmap='Greys')
    ax2 = fig.add_axes([0.51, 0.02, 0.48, 0.96])
    ax2.set_axis_off()
    ax2.imshow(f)


# show output of second conv layer for 5 examples of each digit in exact.pdf
def show_approx_features_chart():
    l = approx_model.layers
    fig = plt.figure(figsize=(8, 8))
    for d in range(10):
        for i in range(5):
            img = random_image(d)
            f = tf.math.sigmoid(l[2](l[1](l[0](img.reshape((1, N, N))))))[0].numpy()
            ax1 = fig.add_axes([0.1 * (2 * i) + 0.001, 0.1 * (9-d) + 0.01, 0.098, 0.098])
            ax1.set_axis_off()
            ax1.imshow(img, cmap='Greys')
            ax2 = fig.add_axes([0.1 * (2 * i + 1) + 0.001, 0.1 * (9-d) + 0.01, 0.098, 0.098])
            ax2.set_axis_off()
            ax2.imshow(f)


def make_approx_model_b(p=10, r=0.01):
    global approx_model_b
    inputs = tf.keras.Input(shape=(N, N))
    reshape = tf.keras.layers.Reshape((N * N,))(inputs)
    hidden = tf.keras.layers.Dense(p)(reshape)
    outputs0 = tf.keras.layers.Dense(10, activation='softmax')(hidden)
    outputs = tf.keras.layers.Reshape((1, 1, 10))(outputs0)
    approx_model_b = tf.keras.Model(inputs=inputs, outputs=outputs, name="approx_model_b")
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    approx_model_b.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return approx_model_b


# try to find training parameters to get perfect performance
EPOCHS = 2000
BATCH_SIZE = 32
r_test = 0.002
all_dataset = tf.data.Dataset.from_tensor_slices((x_all, y_all))
all_dataset = all_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
make_approx_model(r=r_test)
#train_approx_model()
#check_approx_model()
#show_approx_kernels()

weights1 = [[[ 0, -1,  0], [ 1,  1,  1], [ 0, -1,  0]],
            [[ 0, -1,  0], [-1,  1,  1], [ 0, -1,  0]],
            [[ 0, -1,  0], [ 1,  1, -1], [ 0, -1,  0]],
            [[ 0,  1,  0], [-1,  1, -1], [ 0,  1,  0]],
            [[ 0, -1,  0], [-1,  1, -1], [ 0,  1,  0]],
            [[ 0,  1,  0], [-1,  1, -1], [ 0, -1,  0]],
            [[-1, -1,  0], [-1,  1,  1], [ 0,  1, -1]],
            [[ 0, -1, -1], [ 1,  1, -1], [-1,  1,  0]],
            [[ 0,  1, -1], [-1,  1,  1], [-1, -1,  0]],
            [[-1,  1,  0], [ 1,  1, -1], [ 0, -1, -1]],
            [[-1,  1, -1], [-1,  1,  1], [-1,  1, -1]],
            [[-1,  1, -1], [ 1,  1, -1], [-1,  1, -1]]]
bias1 = [-2, -1, -1, -2, -1, -1, -2, -2, -2, -2, -3, -3]

sk = [1,2,10]
weights2 = np.array(weights1)[sk,:].transpose((1, 2, 0)).reshape((3, 3, 1, 3))
bias2 = np.array(bias1)[sk]
#make_approx_model(r=r_test)
#approx_model.layers[2].set_weights([weights2,bias2])
#show_approx_kernels()




from sklearn.decomposition import PCA
x = np.reshape(weights1, (12,9))
np.linalg.matrix_rank(x)
pca = PCA(n_components=6).fit(x)
x1 = np.round(pca.components_, 8)
x2 = x1[0:3,].reshape((3, 3, 1, 3))
b1 = 1-sum(x1[0:3,].T)
pca.explained_variance_ratio_.cumsum()

#approx_model.layers[2].set_weights([x2,b1])

def test_sk():
    sk = [1,2,6]
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0)
    weights1 = [[[ 0, -1,  0], [ 1,  1,  1], [ 0, -1,  0]],
            [[ 0, -1,  0], [-1,  1,  1], [ 0, -1,  0]],
            [[ 0, -1,  0], [ 1,  1, -1], [ 0, -1,  0]],
            [[ 0,  1,  0], [-1,  1, -1], [ 0,  1,  0]],
            [[ 0, -1,  0], [-1,  1, -1], [ 0,  1,  0]],
            [[ 0,  1,  0], [-1,  1, -1], [ 0, -1,  0]],
            [[-1, -1,  0], [-1,  1,  1], [ 0,  1, -1]],
            [[ 0, -1, -1], [ 1,  1, -1], [-1,  1,  0]],
            [[ 0,  1, -1], [-1,  1,  1], [-1, -1,  0]],
            [[-1,  1,  0], [ 1,  1, -1], [ 0, -1, -1]],
            [[-1,  1, -1], [-1,  1,  1], [-1,  1, -1]],
            [[-1,  1, -1], [ 1,  1, -1], [-1,  1, -1]]]
    bias1 = [-2, -1, -1, -2, -1, -1, -2, -2, -2, -2, -3, -3]
    weights2 = np.array(weights1)[sk,:].transpose((1, 2, 0)).reshape((3, 3, 1, 3))
    bias2 = np.array(bias1)[sk]

    for i in range(5):
        make_approx_model(r=r_test)        
        approx_model.layers[2].set_weights([weights2,bias2])
        hist = approx_model.fit(all_dataset,
                                epochs=EPOCHS,
                                verbose=0,
                                callbacks=[callback],
                                validation_data=(x_all, y_all),
                                validation_batch_size=np.shape(y_all)[0])
        print(check_approx_model())
        print(np.shape(hist.history['loss']))
        #show_approx_kernels(mat=0)
        plt.plot(hist.history['val_loss'], label=repr(sk), color='blue')

    for j in range(5):
        make_approx_model(r=r_test)
        hist = approx_model.fit(all_dataset,
                                epochs=EPOCHS,
                                verbose=0,
                                callbacks=[callback],
                                validation_data=(x_all, y_all),
                                validation_batch_size=np.shape(y_all)[0])
        print(check_approx_model())
        print(np.shape(hist.history['loss']))
        #show_approx_kernels(mat=0)
        plt.plot(hist.history['val_loss'], label='random', color='green')
    
    plt.show()

test_sk()
#time series of the loss for plotting etc
#hist.history['loss']

x = np.reshape(weights1, (12,9)).T
y = np.reshape(approx_model.layers[2].get_weights()[0], (3,9)).T

x0 = np.delete(x,(0,3,10,11),1)
y0 = y[:,0]

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x,y)

#hist.history['val_accuracy'][np.shape(hist.history['val_accuracy'])[0]-1]



# big loop


#np.array([combo for combo in combinations(range(12),3)])

#for combo in combinations(lst, 2):  # 2 for pairs, 3 for triplets, etc
#    print(combo)


def test_all():
    global sk_loss_mean, sk_all
    weights1 = [[[ 0, -1,  0], [ 1,  1,  1], [ 0, -1,  0]],
                [[ 0, -1,  0], [-1,  1,  1], [ 0, -1,  0]],
                [[ 0, -1,  0], [ 1,  1, -1], [ 0, -1,  0]],
                [[ 0,  1,  0], [-1,  1, -1], [ 0,  1,  0]],
                [[ 0, -1,  0], [-1,  1, -1], [ 0,  1,  0]],
                [[ 0,  1,  0], [-1,  1, -1], [ 0, -1,  0]],
                [[-1, -1,  0], [-1,  1,  1], [ 0,  1, -1]],
                [[ 0, -1, -1], [ 1,  1, -1], [-1,  1,  0]],
                [[ 0,  1, -1], [-1,  1,  1], [-1, -1,  0]],
                [[-1,  1,  0], [ 1,  1, -1], [ 0, -1, -1]],
                [[-1,  1, -1], [-1,  1,  1], [-1,  1, -1]],
                [[-1,  1, -1], [ 1,  1, -1], [-1,  1, -1]]]
    bias1 = [-2, -1, -1, -2, -1, -1, -2, -2, -2, -2, -3, -3]

    included = [1,2,4,5,6,7,8,9]
    sk_all = np.array([combo for combo in combinations(included,3)])
    #sk_all = [combo for combo in combinations(included,3)]
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0)
    sk_loss_mean = []

    for sk in sk_all: #sk = starting kernels
        weights2 = np.array(weights1)[sk,:].transpose((1, 2, 0)).reshape((3, 3, 1, 3))
        bias2 = np.array(bias1)[sk]
        loss_time = []

        for i in range(1):
            make_approx_model(r=r_test)
            approx_model.layers[2].set_weights([weights2,bias2])
            hist = approx_model.fit(all_dataset,
                                    epochs=EPOCHS,
                                    verbose=0,
                                    callbacks=[callback],
                                    validation_data=(x_all, y_all),
                                    validation_batch_size=np.shape(y_all)[0])
            loss_time.append(hist.history['val_loss'])
    
        loss_mean = np.mean(np.array(loss_time), axis=0)
        sk_loss_mean.append(loss_mean)
        plt.plot(loss_mean, label=repr(sk))
        print(repr(sk)+" done.")

    plt.show()

#test_all()
    