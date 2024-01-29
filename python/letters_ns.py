import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

letters = "iyjcolhtux"

letter_pixels = {
    "i": np.array([[0,1,0],[0,1,0],[0,1,0]]),
    "y": np.array([[1,0,1],[0,1,0],[0,1,0]]),
    "j": np.array([[0,0,1],[0,0,1],[1,1,1]]),
    "c": np.array([[1,1,1],[1,0,0],[1,1,1]]),
    "o": np.array([[1,1,1],[1,0,1],[1,1,1]]),
    "l": np.array([[1,0,0],[1,0,0],[1,1,1]]),
    "h": np.array([[1,0,1],[1,1,1],[1,0,1]]) ,
    "t": np.array([[1,1,1],[0,1,0],[0,1,0]]),
    "u": np.array([[1,0,1],[1,0,1],[1,1,1]]),
    "x": np.array([[1,0,1],[0,1,0],[1,0,1]])
}

def ascii_pixel(c):
    return ('#' if c else ' ')

def ascii_image(l):
    return '\n'.join(map(lambda r: str.join('',map(ascii_pixel,r)),l))

# Choose a letter from tensor output prediction using argmax
def letter_choose(pred):
    i = np.argmax(pred)
    return letters[i] if i < 10 else "REJECT"

# Make a list of all 3x3 images with entries in {0,1} and check
# which ones represent letters from the list above.
all_images = np.array(list(itertools.product(*[[0,1]]*9))).reshape(512,3,3)
all_indices = np.ones(512) * 10
for i in range(10):
    for j in range(512):
        if np.array_equal(all_images[j],letter_pixels[letters[i]]):
            all_indices[j] = i
            break

# Weights and biases for a model that recognises letters in a 
# 3x3 image.  These will be reused for a convolutional layer
# in a later model that recognises letters in a larger grid.
letter_weights = np.array(
    [[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1, 0],
     [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1, 0],
     [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1, 0],
     [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1, 0],
     [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1, 0],
     [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1, 0],
     [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0],
     [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1, 0],
     [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0]])

letter_bias = np.array([-2, -3, -4, -6, -7, -4, -6, -4, -6, -4, 0.5])

# Exact model to recognise letters
# Output can be fed to letter_choose(), essentially adding argmax
# activation to the final layer
def make_exact_letter_model():
    """
    Make an exact model that recognises letters from 3x3 images.

    The input is a 3x3 image with entries in {0,1}.  The output is a
    vector of length 11, where the first 10 entries are the probabilities
    that the image is a particular letter from the string letters
    defined above.  The last entry is the probability that the image 
    is not a letter.  These probabilities will always be close to 0 or 1,
    so we can use argmax to choose the letter.
    """
    inputs = tf.keras.Input(shape=(3, 3))
    reshape = tf.keras.layers.Reshape((9,))(inputs)
    outputs = tf.keras.layers.Dense(11, activation='relu')(reshape)
    M = tf.keras.Model(inputs=inputs, outputs=outputs, name="explicit_model")
    M.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    M.layers[2].set_weights([letter_weights, letter_bias])
    return M

def test_letter_model(M = None):
    if M is None:
        M = make_exact_letter_model()
    pred = np.argmax(M.predict(all_images,verbose=0), axis=1)
    return np.array_equal(pred, all_indices)

def write_letter(grid, letter, x, y):
    """
    Write a letter into a grid at position (x,y)

    The grid is expected to be a numpy array of shape (h,w) say.
    The letter is expected to be a string from the string letters
    defined above.  The position (x,y) is expected to be a pair of 
    integers with 0 <= x < w-2 and 0 <= y < h-2.  Note that the 
    horizontal coordinate is given first, and the vertical coordinate 
    counts down from the top of the grid.
    """
    h, w = np.shape(grid)
    if y > h - 3 or x > w - 3:
        print(f"In grid of width {w} and height {h}, letter cannot be added with  (x,y)=({x},{y})")
        return grid
    elif not letter in letters:
        print(f"Letter {letter} not recognised")
        return grid
    else:
        grid[y:y+3,x:x+3] = letter_pixels[letter]
        return grid

def make_exact_grid_model(width=15, height=6):
    """
    Make an exact model that counts letters in a grid of the specified size.

    The input is a grid of size (height, width) with entries in {0,1}.
    The output is a vector of length 11, where the first 10 entries are the
    numbers of instances of the letters in the string letters defined above,
    and the last entry is the number of 3x3 blocks that do not contain a letter.
    """
    inputs = tf.keras.Input(shape=(height, width)) # add function arguments for this
    reshape = tf.keras.layers.Reshape((height, width, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(11, [3, 3], [1, 1], padding='valid', activation='linear')(reshape)
    rescale = tf.keras.layers.Rescaling(scale=10)(convlayer)
    softmax = tf.keras.layers.Softmax(axis=3)(rescale)
    straighten = tf.keras.layers.Reshape(((height - 2) * (width - 2), 11))(softmax)
    sum = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(straighten)
    M = tf.keras.Model(inputs=inputs, outputs=sum, name="exact_grid_model")
    M.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    M.layers[2].set_weights([letter_weights.reshape(3,3,1,11), letter_bias])
    return M

def make_ragged_grid(word, width=None, height=6):
    """
    Make a grid with a word written in it.
    
    The word is expected to be a string from the string letters defined above.
    If the width is not specified, it is taken to be 5 times the length of the word.
    The letters are placed in the grid in a roughly uniform way, but with some
    perturbation both horizontally and vertically.
    """
    n = len(word)
    if width is None:
        width = 5*n
    grid = np.zeros([height,width])
    if n == 0:
        return grid
    for i, c in enumerate(word):
        x_pos = np.random.randint(np.round((i*width)/n), np.round(((i+1)*width)/n-3))
        y_pos = np.random.randint(0,height-3)
        grid = write_letter(grid,word[i],x_pos,y_pos)
    return grid

def test_grid_model(word, M = None, grid = None, width=None, height=6):
    n = len(word)
    if grid is None:
        if width is None:
            width = 5*n
        grid = make_ragged_grid(word, width, height)
    width, height = np.shape(grid)
    if M is None:
        M = make_exact_grid_model(width, height)
    wc0 = [word.count(c) for c in letters]
    wc0.append((height-2) * (width-2) - np.sum(wc0))
    wc0 = np.array(wc0) 
    print(wc0)
    wc0 = [word.count(c) for c in letters]
    wc1 = M.predict(grid.reshape(1,height,width),verbose=0)[0]
    wc1 = np.round(wc1).astype(int)
    return np.array([wc0, wc1])

# Now suppose we want to make a model that can recognise words in a grid.
# Using the same approach as above, we can make a model that looks in 
# each vertical band of width 3 and returns a code for the letter 
# found in that band, or zero if there is no letter in that band.
# To go further, we want to strip out the zeros.  Below we define 
# two models that do roughly this, but with different details.

def make_zero_filter(p):
    """
    Make a model that returns the last p nonzero values in a sequence x,
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
    inputs = tf.keras.Input(shape=(None,1))
    nz = tf.keras.layers.Dense(2, activation='relu')(inputs)
    gru = tf.keras.layers.GRU(p, activation='linear', recurrent_activation='linear', reset_after=False)(nz)
    M = tf.keras.Model(inputs=inputs, outputs=gru, name='zero_filter')
    M.compile(loss='mean_squared_error', metrics=['accuracy'])
    w1 = np.array([[1,-1]])
    b1 = np.array([0,1])
    M.layers[1].set_weights([w1, b1])
    v2 = np.zeros((2,3*p))
    v2[0,3*p-1] = 1
    v2[1,0:p] = np.ones(p)
    w2 = np.zeros((p,3*p))
    w2[1:p,2*p:3*p-1] = np.eye(p-1)
    b2 = np.zeros(3*p)
    b2[p:2*p] = np.ones(p)
    M.layers[2].set_weights([v2, w2, b2])
    return M
    
def make_zero_filter_alt(p):
    """Make a model that takes a p-vector x and moves any zeros to the end.

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
    inputs = tf.keras.Input(shape=(p,1))
    nz = tf.keras.layers.Dense(2, activation='relu')(inputs)
    h0 = np.zeros((1,3*p))
    h0[0,0] = 1
    h0[0,p] = 1
    kh0 = tf.keras.backend.constant(h0)
    gru = tf.keras.layers.GRU(3*p, activation='linear', 
                              recurrent_activation='linear', 
                              reset_after=False)(nz, initial_state=kh0)
    M = tf.keras.Model(inputs=inputs, outputs=gru, name='zero_filter')
    M.compile(loss='mean_squared_error', metrics=['accuracy'])
    w1 = np.array([[1,-1]])
    b1 = np.array([0,1])
    M.layers[1].set_weights([w1, b1])
    v2 = np.zeros((2,9*p))
    v2[0,4*p:5*p] = np.ones(p)
    v2[1,0:3*p] = np.ones(3*p)
    w2 = np.zeros((3*p,9*p))
    w2[0:p-1,6*p+1:7*p] = np.eye(p-1)
    w2[0:p-1,7*p+1:8*p] = np.eye(p-1)
    w2[p:2*p,8*p:9*p] = np.eye(p)
    w2[2*p:3*p,8*p:9*p] = np.eye(p)
    b2 = np.zeros(9*p)
    b2[3*p:4*p] = np.ones(p)
    b2[5*p:6*p] = np.ones(p)
    M.layers[2].set_weights([v2, w2, b2])
    return M

def gru_model(p, q):
    grid_l = int(np.floor(q/3))
    inputs = tf.keras.Input(shape=(p, q))
    reshape1 = tf.keras.layers.Reshape((p, q, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(11, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(p-1,3), strides=(p-1,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Reshape((grid_l, 11))(pooling)
    
    # Recurrent1 converts the convolution output to a sequence of integers
    recurrent1 = tf.keras.layers.SimpleRNN(1, return_sequences=True, activation='linear')(reshape2)
    
    # Dense1 and dense2 produce I(x>0)
    dense1 = tf.keras.layers.Dense(2, activation='relu')(recurrent1)
    dense2 = tf.keras.layers.Dense(1, activation='linear')(dense1)

    # Recurrent2 produces increments at non-zeros
    recurrent2 = tf.keras.layers.GRU(1, activation='linear', recurrent_activation='linear', return_sequences=True)(dense2)
    
    # Dense2 and dense3 produce I(s=j), for j = {1,...,10}
    dense3 = tf.keras.layers.Dense(3*grid_l, activation='relu')(recurrent2)
    dense4 = tf.keras.layers.Dense(grid_l, activation='linear')(dense3)

    # Merge dense layers and compute z using 2 layer bool
    # problem atm is that if I(s=j)=0 and I(x<=0)=1 then z=2 so everything doubles instead of staying the same
    # AND can be calculated as I(s=j)*I(x>0) = relu(I(s=j)+I(x>0)-1) in one dense layer
    merge1 = tf.keras.layers.Concatenate(axis=-1)([dense2,dense4])
    dense5 = tf.keras.layers.Dense(grid_l, activation='relu')(merge1)

    # Recurrent3 returns the original sequence with the non-zeros first
    merge2 = tf.keras.layers.Concatenate(axis=-1)([recurrent1,dense5])
    recurrent3 = tf.keras.layers.GRU(grid_l, activation='linear', recurrent_activation='relu', return_sequences=True, reset_after=False)(merge2)
    
    M = tf.keras.Model(inputs=inputs, outputs=recurrent3, name='gru_model')
#    M = tf.keras.Model(inputs=inputs, outputs=dense3, name='gru_model')
    M.compile(loss='mean_squared_error', metrics=['accuracy'])

    # Convolution layer
    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1, 0],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1, 0],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1, 0],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1, 0],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1, 0],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1, 0],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1, 0],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0]])
    
    # Turns out we can just reshape w0 from the first model
    w1 = w0.reshape(3,3,1,11)

    b0 = np.array([-2.5, -3.5, -4.5, -6.5, -7.5, -4.5, -6.5, -4.5, -6.5, -4.5, 0])

    M.layers[2].set_weights([2*w1, 2*b0])

    # Recurrent layer 1 converts the one-hot input to integers
    w2 = np.arange(1,12).reshape(11,1)
    w3 = np.zeros((1,1))
    b1 = np.zeros(1)
    M.layers[5].set_weights([w2, w3, b1])

    # Dense1
    w4 = np.ones((1,2))
    b2 = np.array([0,-1])
    M.layers[6].set_weights([w4, b2])

    # Dense2
    w5 = np.array([[1],[-1]])
    b3 = np.zeros(1)
    M.layers[7].set_weights([w5, b3])

    # Recurrent layer 2 increments by 1 at each unmasked value
    w6 = np.array([[-1,0,0]]) #z_t = I(x=<0) = 1 - I(x>0)
    w7 = np.array([[0,0,1]]) #h^t = h_t-1 + 1
    b4 = np.array([[1,1,1],[0,0,0]]) #r_t = 1, h^t = h_t-1 + 1
    M.layers[8].set_weights([w6, w7, b4])

    #dense3
    # need 1+s-j, s-j, s-1-j
    # have as (1+s-1, s-1, s-1-1, 1+s-2, s-2, s-1-2,...)
    w8 = np.ones((1,3*grid_l))
    b5 = np.repeat(-np.arange(grid_l),3)
    b5[1::3] = -np.arange(grid_l)-1
    b5[2::3] = -np.arange(grid_l)-2
    M.layers[9].set_weights([w8, b5])

    #dense4
    # need to calculate this before adding bias in the following GRU layer
    w9 = np.zeros((3*grid_l,grid_l))
    for j in range(grid_l):
        w9[(j*3):((j+1)*3),j] = np.array([1,-2,1])
    b6 = np.zeros(grid_l)
    M.layers[10].set_weights([w9, b6])

    #dense5
    # calculate I(s=j and x>0)
    w10 = np.zeros((1+grid_l,grid_l))
    w10[0,:] = np.ones(grid_l)
    w10[1:(1+grid_l),:] = np.eye(grid_l)
    b7 = -np.ones(grid_l)
    M.layers[12].set_weights([w10, b7])

    #recurrent3
    # layer receives:
    # x_t
    # I(x<=0)
    # 10 parts for I(s=j)
    w11 = np.zeros((1+grid_l, 3*grid_l))
    # h_t = x_t
    w11[0,(2*grid_l):(3*grid_l)] = np.ones(grid_l)
    # z_t = 1 - I(s=j and x>0)
    w11[1:(1+grid_l),0:grid_l] = -np.eye(grid_l)

    w12 = np.zeros((grid_l,3*grid_l))

    b8 = np.zeros(3*grid_l)
    b8[0:grid_l] = np.ones(grid_l)
    M.layers[14].set_weights([w11, w12, b8])

    return M
    # The GRU layer has one unit for each position
    # Each unit is biased so that the recurrent2 input switches
    # the forget and reset gates for a particular unmasked value
    # layer[9]

    # np.shape(lstm_model.layers[9].get_weights()[0])
    # (2, 30) = (input shape , units x3)

    # np.shape(lstm_model.layers[9].get_weights()[1])
    # (10, 30) = (units, units x3)

    # np.shape(lstm_model.layers[9].get_weights()[2])
    # (2, 30) = (input shape , units x3)
