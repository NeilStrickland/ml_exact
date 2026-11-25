import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os


src_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.realpath(os.path.join(src_dir, '../models'))


def convert_to_input(x):
    if x == 32:
        return [1,0,0,0,0,0,0] # space
    elif x == 122:
        return [0,1,0,0,0,0,0] # z
    elif x == 101:
        return [0,0,1,0,0,0,0] # e
    elif x == 114:
        return [0,0,0,1,0,0,0] # r
    elif x == 111:
        return [0,0,0,0,1,0,0] # o
    elif x == 110:
        return [0,0,0,0,0,1,0] # n
    else:
        return [0,0,0,0,0,0,1] # anything else

# [122,101,114,111] = zero
# [111,110,101] = one

#text = 'one and zero and one and zero with one not zero but zero not one'
def lstm_attempt():
    file_path = os.path.join(src_dir, '..\\notes\\zara.txt')

    with open(file_path, 'r') as file:
        text = file.read()

    text_num = [ord(x) for x in list(text)]


    text_flag = [0]*len(text_num)
    for i in range(0, len(text_num)):
        if text_num[i-3:i+1] == [122,101,114,111]:
            text_flag[i] = 0.5
        elif text_num[i-2:i+1] == [111,110,101]:
            text_flag[i] = 1
        else:
            text_flag[i] = 0

#print(text_num)
#print(text_flag)

    text_num = [ord(x)/122 for x in list(text)]
    x_train = np.array(text_num)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, 1))
    y_train = np.array(text_flag)

# Create the RNN model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=(1,1)))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

# Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=500)



    test_text = 'one and zero.'
    test_text_num = [ord(x)/122 for x in list(test_text)]
    x_test = np.array(test_text_num)
    x_test = np.reshape(x_test, (x_test.shape[0], 1, 1))

    predictions = model.predict(x_test)

    print(predictions)



def state_rnn_attempt():
    # prepare input as nx6 array where we have each element as
    # [z, e, r, o, n, [anything else]] as 0/1s

    file_path = os.path.join(src_dir, '..\\notes\\zara.txt')

    with open(file_path, 'r') as file:
        text = file.read()

    text = 'one zero and one and zero'

    text_num = [ord(x) for x in list(text)]

    text_conv = list(map(convert_to_input, text_num))
    text_tensor = tf.convert_to_tensor(text_conv, dtype=tf.float32)
    text_reshape = tf.reshape(text_tensor, [1, text_tensor.shape[0], 7])

    # We have 10 possible states
    # The state vector will be initialised as all 0s; this is the base state
    # The other 9 states are in the vector:
    # [z, ze, zer, zero, o, on, one, out0, out1]

    # The rules of input + state we want for the states are:
    # z + base -> z
    # e + z -> ze
    # r + ze -> zer
    # o + zer -> zero
    # space + zero -> out0
    # o + base -> o
    # n + o -> on
    # e + on -> one
    # space + one -> out1
    # anything else + any state -> base

    # and for the output:
    # out0 -> zero
    # out1 -> one
    # anything other state -> nothing

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(9, return_sequences=True, activation='relu'))

    # n = dimension of input
    # m = units argument to SimpleRNN (number of states?)

    # model.get_weights()
    # model.get_weights()[0].shape # (n, m)
    # model.get_weights()[1].shape # (m, m)
    # model.get_weights()[2].shape # (m,)

    # do we add specific output states zero and one?
    # they could have same state rules as base but direct outputs

    # below adds a dense layer to reduce the output to 3 numbers
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    test = model.predict(text_reshape)
    # model.get_weights()[3].shape # (m, 3)

    W00 = np.array([[0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0,-1,-1,-1,-1,-1,-1],
                    [0,-1,-1,-1,-1,-1,-1]])

    W01 = np.array([[-1,-1,-1,-1,-1,-1,-1, 0, 0],
                    [ 0,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1, 0,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1, 0,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1,-1, 0, 0],
                    [-1,-1,-1,-1, 0,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1, 0,-1,-1,-1],
                    [-1,-1,-1, 1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1, 1,-1,-1]])

    B0 = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0])

    W10 = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0, 0,20, 0],
                    [ 0, 0, 0, 0, 0, 0, 0, 0,20]])

    B1 = np.array([ 0,-10,-10])

    model.layers[0].set_weights([W00.T, W01.T, B0.T])
    model.layers[1].set_weights([W10.T, B1.T])


    test = model.predict(text_reshape)


# Start again with functions to tweak the model

#file_path = os.path.join(src_dir, '..\\notes\\zara.txt')

#with open(file_path, 'r') as file:
#    text = file.read()

text = 'one zero anone one and zero '

text_num = [ord(x) for x in list(text)]
  
text_conv = list(map(convert_to_input, text_num))
text_tensor = tf.convert_to_tensor(text_conv, dtype=tf.float32)
text_reshape = tf.reshape(text_tensor, [1, text_tensor.shape[0], 7])


# Test model for manually adjusting weights and layers
def test_model(text_tensor):

    # Create a model and run an arbitrary prediction to initialize
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(10, return_sequences=True, activation='relu'))
    test = model.predict(text_tensor)

    # new state = relu(w0*input + w1*current state + b0)

    W0 = np.array([[ 0, 1, 0, 0, 0, 0, 0],
                   [ 0, 0, 1, 0, 0, 0, 0],
                   [ 0, 0, 0, 1, 0, 0, 0],
                   [ 0, 0, 0, 0, 1, 0, 0],
                   [ 0, 0, 0, 0, 1, 0, 0],
                   [ 0, 0, 0, 0, 0, 1, 0],
                   [ 0, 0, 1, 0, 0, 0, 0],
                   [ 1, 0, 0, 0, 0, 0, 0],
                   [ 1, 0, 0, 0, 0, 0, 0],
                   [-1, 1, 1, 1, 1, 1, 1]])



    #                   [in this state]
    W1 = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                   [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 0,-1, 0, 0, 0, 0, 0, 0,-1], # [add to this state]
                   [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

    B0 = np.array([ 0, -1, -1, -1, 0, -1, -1, -1, -1, 0])

    model.layers[0].set_weights([W0.T, W1.T, B0.T])

    output = model.predict(text_tensor)
    print(output)


#test_model(text_tensor=text_reshape)


# Through the test model we found that we need a separate 'block switch'
# This blocks 'z' or 'o' states unless a space has occurred
# Space turns off the block switch, any other input turns it back on



# Create two functions:
# One with the model output being the states
# One with the model output being [*, zero, one]

def state_model(text_tensor):

    # Create a model and run an arbitrary prediction to initialize
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(10, return_sequences=True, activation='relu'))
    test = model.predict(text_tensor)

    W0 = np.array([[ 0, 1, 0, 0, 0, 0, 0],
                   [ 0, 0, 1, 0, 0, 0, 0],
                   [ 0, 0, 0, 1, 0, 0, 0],
                   [ 0, 0, 0, 0, 1, 0, 0],
                   [ 0, 0, 0, 0, 1, 0, 0],
                   [ 0, 0, 0, 0, 0, 1, 0],
                   [ 0, 0, 1, 0, 0, 0, 0],
                   [ 1, 0, 0, 0, 0, 0, 0],
                   [ 1, 0, 0, 0, 0, 0, 0],
                   [-1, 1, 1, 1, 1, 1, 1]])

    W1 = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                   [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                   [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    B0 = np.array([ 0, -1, -1, -1, 0, -1, -1, -1, -1, 0])

    model.layers[0].set_weights([W0.T, W1.T, B0.T])

    output = model.predict(text_tensor)
    print(output)


#state_model(text_tensor=text_reshape)




def full_model(text_tensor):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(10, return_sequences=True, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    test = model.predict(text_tensor)

    W00 = np.array([[ 0, 1, 0, 0, 0, 0, 0],
                    [ 0, 0, 1, 0, 0, 0, 0],
                    [ 0, 0, 0, 1, 0, 0, 0],
                    [ 0, 0, 0, 0, 1, 0, 0],
                    [ 0, 0, 0, 0, 1, 0, 0],
                    [ 0, 0, 0, 0, 0, 1, 0],
                    [ 0, 0, 1, 0, 0, 0, 0],
                    [ 1, 0, 0, 0, 0, 0, 0],
                    [ 1, 0, 0, 0, 0, 0, 0],
                    [-1, 1, 1, 1, 1, 1, 1]])

    W01 = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                    [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [ 0, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                    [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    B0 = np.array([ 0, -1, -1, -1, 0, -1, -1, -1, -1, 0])

    W10 = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0, 0,20, 0, 0],
                    [ 0, 0, 0, 0, 0, 0, 0, 0,20, 0]])

    B1 = np.array([ 0,-10,-10])

    model.layers[0].set_weights([W00.T, W01.T, B0.T])
    model.layers[1].set_weights([W10.T, B1.T])

    output = model.predict(text_tensor)
    print(output)

full_model(text_tensor=text_reshape)

