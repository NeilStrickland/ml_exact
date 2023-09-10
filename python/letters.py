# Try example 6.5
import tensorflow as tf
import numpy as np


# Generate a "pixel" array from input letter
def letter_input(letter):
    match letter:
        case "i": return np.array([
            [0,1,0],
            [0,1,0],
            [0,1,0]
        ])
        case "y": return np.array([
            [1,0,1],
            [0,1,0],
            [0,1,0]
        ])
        case "j": return np.array([
            [0,0,1],
            [0,0,1],
            [1,1,1]
        ])
        case "c": return np.array([
            [1,1,1],
            [1,0,0],
            [1,1,1]
        ])
        case "o": return np.array([
            [1,1,1],
            [1,0,1],
            [1,1,1]
        ])
        case "l": return np.array([
            [1,0,0],
            [1,0,0],
            [1,1,1]
        ])
        case "h": return np.array([
            [1,0,1],
            [1,1,1],
            [1,0,1]
        ]) 
        case "t": return np.array([
            [1,1,1],
            [0,1,0],
            [0,1,0]
        ])
        case "u": return np.array([
            [1,0,1],
            [1,0,1],
            [1,1,1]
        ])
        case "x": return np.array([
            [1,0,1],
            [0,1,0],
            [1,0,1]
        ])


# Choose a letter from tensor output prediction using argmax
def letter_choose(pred):
    i = np.argmax(pred)
    match i:
        case 0:  return "i"
        case 1:  return "y"
        case 2:  return "j"
        case 3:  return "c"
        case 4:  return "o"
        case 5:  return "l"
        case 6:  return "h"
        case 7:  return "t"
        case 8:  return "u"
        case 9:  return "x"
        case _: return "REJECT"


# Explicit model to recognise letters
# Output can be fed to letter_choose(), essentially adding argmax
# activation to the final layer
def explicit_model():
    global explicit_model
    inputs = tf.keras.Input(shape=(3, 3))
    reshape = tf.keras.layers.Reshape((9,))(inputs)
    outputs = tf.keras.layers.Dense(11, activation='linear')(reshape)
    explicit_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="explicit_model")
    explicit_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1, 0],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1, 0],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1, 0],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1, 0],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1, 0],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1, 0],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1, 0],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0]])

    b0 = np.array([-2.5, -3.5, -4.5, -6.5, -7.5, -4.5, -6.5, -4.5, -6.5, -4.5, 0])

    explicit_model.layers[2].set_weights([w0, b0])
    return explicit_model

explicit_model()

#explicit_model.predict(letter_input("i").reshape(1, 3, 3), verbose=0)[0]
#letter_choose(explicit_model.predict(letter_input("i").reshape(1, 3, 3),verbose=0)[0])

# Test all letters and one non-letter
def test_explicit_model():
    for letter in ["i", "y", "j", "c", "o", "l", "h", "t", "u", "x"]:
        print(letter_choose(explicit_model.predict(letter_input(letter).reshape(1, 3, 3),verbose=0)[0]))

    print(letter_choose(explicit_model.predict(np.array([[1, 1, 0],
                                                         [0, 1, 0],
                                                         [0, 1, 0]]).reshape(1, 3, 3),verbose=0)[0]))


test_explicit_model()

# Convert this to convolution network
# The idea is to take the weights and turn them into 9 separate 3x3 convolution
# channels


# add a letter to a grid with top left i,j
def write_letter(grid, letter, i, j):
    if i > np.shape(grid)[0] - 3 or j > np.shape(grid)[1] - 3:
        print("Letter cannot be added with top corner" + repr(i) + ", " + repr(j))
        return grid
    else:
        grid[i:i+3,j:j+3] = letter_input(letter)
        return grid
    

# Fix to 6x15 grid with 4 letter words for development
zero_grid = np.zeros([6,15])
input_grid = write_letter(zero_grid,"c",0,0)
input_grid = write_letter(input_grid,"i",1,4)
input_grid = write_letter(input_grid,"t",2,8)
input_grid = write_letter(input_grid,"y",3,12)


# 11 channels, 1 per letter (inluding reject)
# 3x3 filter shape
# strides of 1 horizontally, 1 vertically

# for max pooling use the height of the input-2, 3-wide, stride 3 across
# to find each letter detetced once

# then reshape to one-hot encoding and pass into rnn?
def conv_model():
    global conv_model
    inputs = tf.keras.Input(shape=(6, 15)) # add function arguments for this
    reshape = tf.keras.layers.Reshape((6, 15, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(11, [3, 3], [1, 1], 'valid', activation='linear')(reshape)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(4,3), strides=(1,3))(convlayer)
    conv_model = tf.keras.Model(inputs=inputs, outputs=pooling, name="approx_model")
    conv_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

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

    conv_model.layers[2].set_weights([w1, b0])

    return conv_model

conv_model()

np.shape(conv_model.predict(input_grid.reshape(1,6,15),verbose=0))
conv_model.predict(input_grid.reshape(1,6,15),verbose=0)[0,:,:,0] # first channel output (i)
conv_model.predict(input_grid.reshape(1,6,15),verbose=0)[0,:,:,3] # fourth channel output (c)
np.shape(conv_model.layers[2].get_weights()[0]) # (3,3,1,10) can reshape to (3,3,10)
np.shape(conv_model.layers[2].get_weights()[1]) # (10,)