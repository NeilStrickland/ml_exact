# Try example 6.5
import tensorflow as tf
import numpy as np
import os


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
# Strides of 1 horizontally, 1 vertically

# For max pooling use the height of the input-1, 3-wide, stride 3 across
# to find each letter detected once
# Stride height of input-1 limits the output to 1 row
def conv_model():
    global conv_model
    inputs = tf.keras.Input(shape=(6, 15)) # add function arguments for this
    reshape = tf.keras.layers.Reshape((6, 15, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(11, [3, 3], [1, 1], padding='valid', activation='linear')(reshape)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(5,3), strides=(5,3), padding='same')(convlayer)
    conv_model = tf.keras.Model(inputs=inputs, outputs=pooling, name="conv_model")
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
np.shape(conv_model.layers[2].get_weights()[0]) # (3,3,1,11) can reshape to (3,3,11)
np.shape(conv_model.layers[2].get_weights()[1]) # (11,)


# Test the convolution model on an input word
# p is the number of rows; the letters are scattered at random heights
# q is the number of columns; the letters are scattered roughly evenly over the rows
# For now fix to p=6, q=15
def test_conv_model(p, q, word):
    word_list = list(word)
    n = len(word_list) 
    if p < 3:
        raise ValueError("p must be at least 3 to fit the word in.")
    if q/n < 3:
        raise ValueError("q must be at least 3*word-length to fit the word in.")
    
    input_grid = np.zeros([p,q])
    for i in range(n):
        x_pos = np.random.randint(0,p-3)
        y_pos = np.random.randint(i*np.round(q/n), (i+1)*np.round(q/n)-3)
        input_grid = write_letter(input_grid,word_list[i],x_pos,y_pos)
    print(input_grid)

    pred = conv_model.predict(input_grid.reshape(1,p,q),verbose=0)
    m = np.shape(pred)[2]
    for j in range(m):
       print(letter_choose(pred[0,0,j,:]))

test_conv_model(6, 15, "city")

# then reshape to one-hot encoding and pass into rnn?




# The example notes suggest two extensions
# The first is keeping a running total of the counts of each letter
# Add relu(2x) to the conv layer from above which results in one-hot encoding for the letters
# Pass into recurrent layer with 0 bias and identity weights to simply add each input
# into the memory
def sum_model():
    global sum_model
    inputs = tf.keras.Input(shape=(6, 15))
    reshape1 = tf.keras.layers.Reshape((6, 15, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(11, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(5,3), strides=(5,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Reshape((5, 11))(pooling)
    recurrent = tf.keras.layers.SimpleRNN(11, return_sequences=False, activation='relu')(reshape2)
    sum_model = tf.keras.Model(inputs=inputs, outputs=recurrent, name="sum_model")
    sum_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    # Convolution layer - as above
    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1, 0],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1, 0],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1, 0],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1, 0],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1, 0],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1, 0],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1, 0],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0]])
    
    w1 = w0.reshape(3,3,1,11)

    b0 = np.array([-2.5, -3.5, -4.5, -6.5, -7.5, -4.5, -6.5, -4.5, -6.5, -4.5, 0])

    sum_model.layers[2].set_weights([2*w1, 2*b0]) # Add 2*

    w2 = np.eye(11)

    b1 = np.zeros(11)

    sum_model.layers[5].set_weights([w2, w2, b1])

    return sum_model

sum_model()


np.shape(sum_model.predict(input_grid.reshape(1,6,15),verbose=0))
sum_model.predict(input_grid.reshape(1,6,15),verbose=0)[0,0] # first channel output (i)
np.shape(sum_model.layers[5].get_weights()[0]) # (11,11)
np.shape(sum_model.layers[5].get_weights()[1]) # (11,11)
np.shape(sum_model.layers[5].get_weights()[2]) # (11,)


# Make data for training the word recognition part of the model
# using the outputs from the existing model
def write_word(p,q,word):
    word_list = list(word)
    n = len(word_list) 
    if p < 3:
        raise ValueError("p must be at least 3 to fit the word in.")
    if q/n < 3:
        raise ValueError("q must be at least 3*word-length to fit the word in.")
    
    input_grid = np.zeros([p,q])

    for i in range(n):
        x_pos = np.random.randint(0,p-3)
        y_pos = np.random.randint(i*np.round(q/n), (i+1)*np.round(q/n)-3)
        input_grid = write_letter(input_grid,word_list[i],x_pos,y_pos)
    return input_grid


# As we fix grid shape we can do sums with dense layer

# The permute layer changes the max pooling output from
# 10 outputs with 11 channels to 11 outputs (one for each letter)
# over 10 channels
# The channels are then separately summed into 1 dense output each
def pq_conv_model(p, q):
    global pq_conv_model
    inputs = tf.keras.Input(shape=(p, q))
    reshape1 = tf.keras.layers.Reshape((p, q, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(11, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(p-1,3), strides=(p-1,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Permute((1,3,2))(pooling)
    sum = tf.keras.layers.Dense(1, activation='linear')(reshape2)
    pq_conv_model = tf.keras.Model(inputs=inputs, outputs=sum, name='pq_conv_model')
    pq_conv_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

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

    pq_conv_model.layers[2].set_weights([2*w1, 2*b0])

    w2 = np.ones([10,1])

    b1 = np.zeros(1)

    pq_conv_model.layers[5].set_weights([w2, b1])




zero_grid = np.zeros([6,30])
input_grid = write_letter(zero_grid,"c",0,0)
input_grid = write_letter(input_grid,"c",1,4)
input_grid = write_letter(input_grid,"c",2,8)
input_grid = write_letter(input_grid,"y",3,12)

   

pq_conv_model(p=6,q=30)

np.shape(pq_conv_model.predict(input_grid.reshape(1,6,30),verbose=0)) #(1,1,q/3,1)
#np.shape(pq_conv_model.layers[5].get_weights()[0]) # (11,1)
#np.shape(pq_conv_model.layers[5].get_weights()[1]) # (1,)
pq_conv_model.predict(input_grid.reshape(1,6,30),verbose=0)



# can we make model using the sums and character positional encoding only?

# masking means that the output rows with all 0s are ignored by the lstm layer


# Idea:
# Set the rnn units to the max length of the words
# Get the ith entry of the final rnn output to be the letter coded as 1-10
# (this is the alternative to one-hot embedding)
# eg. "city" would be (3,1,4,2,0,0,0,0,0,0), "cit" would be (3,1,4,0,0,0,0,0,0,0)

# may need LSTM or extra RNN channel to get this to work?
# or some kind of "shifter" matrix to send say [1,0, ...] to [0,1, ...] etc


# after this:
# can use StringLookup layer with invert parameter to handle the mapping back to characters/words
def gru_model(p, q):
    global gru_model
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
    recurrent3 = tf.keras.layers.GRU(grid_l, activation='linear', recurrent_activation='relu', return_sequences=False, reset_after=False)(merge2)
    
    gru_model = tf.keras.Model(inputs=inputs, outputs=recurrent3, name='gru_model')
    gru_model.compile(loss='mean_squared_error', metrics=['accuracy'])

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

    gru_model.layers[2].set_weights([2*w1, 2*b0])

    # Recurrent layer 1 converts the one-hot input to integers
    w2 = np.arange(1,12).reshape(11,1)
    w3 = np.zeros((1,1))
    b1 = np.zeros(1)
    gru_model.layers[5].set_weights([w2, w3, b1])

    # Dense1
    w4 = np.ones((1,2))
    b2 = np.array([0,-1])
    gru_model.layers[6].set_weights([w4, b2])

    # Dense2
    w5 = np.array([[1],[-1]])
    b3 = np.zeros(1)
    gru_model.layers[7].set_weights([w5, b3])

    # Recurrent layer 2 increments by 1 at each unmasked value
    w6 = np.array([[-1,0,0]]) #z_t = I(x=<0) = 1 - I(x>0)
    w7 = np.array([[0,0,1]]) #h^t = h_t-1 + 1
    b4 = np.array([[1,1,1],[0,0,0]]) #r_t = 1, h^t = h_t-1 + 1
    gru_model.layers[8].set_weights([w6, w7, b4])

    #dense3
    # need 1+s-j, s-j, s-1-j
    # have as (1+s-1, s-1, s-1-1, 1+s-2, s-2, s-1-2,...)
    w8 = np.ones((1,3*grid_l))
    b5 = np.repeat(-np.arange(grid_l),3)
    b5[1::3] = -np.arange(grid_l)-1
    b5[2::3] = -np.arange(grid_l)-2
    gru_model.layers[9].set_weights([w8, b5])

    #dense4
    # need to calculate this before adding bias in the following GRU layer
    w9 = np.zeros((3*grid_l,grid_l))
    for j in range(grid_l):
        w9[(j*3):((j+1)*3),j] = np.array([1,-2,1])
    b6 = np.zeros(grid_l)
    gru_model.layers[10].set_weights([w9, b6])

    #dense5
    # calculate I(s=j and x>0)
    w10 = np.zeros((1+grid_l,grid_l))
    w10[0,:] = np.ones(grid_l)
    w10[1:(1+grid_l),:] = np.eye(grid_l)
    b7 = -np.ones(grid_l)
    gru_model.layers[12].set_weights([w10, b7])

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
    gru_model.layers[14].set_weights([w11, w12, b8])


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


zero_grid = np.zeros([6,20])
input_grid = write_letter(zero_grid,"c",0,0)
input_grid = write_letter(input_grid,"i",1,4)
input_grid = write_letter(input_grid,"t",2,8)
input_grid = write_letter(input_grid,"x",3,17)

   

gru_model(p=6,q=20)

#np.shape(lstm_model.predict(input_grid.reshape(1,6,30),verbose=0)) #(1,1,q/3,11)
#np.shape(lstm_model.layers[5].get_weights()[0]) # (11,1)
#np.shape(lstm_model.layers[5].get_weights()[1]) # (1,)
gru_model.predict(input_grid.reshape(1,6,20),verbose=0)


# from here we can convert the sequence into an integer and use a
# string lookup layer to complete the word detection

# to convert the sequence use a dense layer with 1 output and weights 11^j

def make_vocab(file):
    src_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(src_dir, file)
    words_3x3 = [l.replace('\n', '').lower() for l in open(file_path, 'r').readlines()]
    letters = ["i","y","j","c","o","l","h","t","u","x"]
    layer1 = tf.keras.layers.StringLookup(vocabulary=letters)
    numbers_3x3 = []
    for word in words_3x3:
        digits = layer1(list(word)).numpy()
        word_sum = sum(digits*np.array([11]*len(digits))**np.arange(len(digits)))
        numbers_3x3.append(word_sum)
    return words_3x3, numbers_3x3

#words_3x3, numbers_3x3 = make_vocab(file='..\\notes\\words_3x3.txt')

def full_model(p, q):
    global full_model
    grid_l = int(np.floor(q/3))
    inputs = tf.keras.Input(shape=(p, q))
    reshape1 = tf.keras.layers.Reshape((p, q, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(11, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(p-1,3), strides=(p-1,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Reshape((grid_l, 11))(pooling)
    recurrent1 = tf.keras.layers.SimpleRNN(1, return_sequences=True, activation='linear')(reshape2)
    dense1 = tf.keras.layers.Dense(2, activation='relu')(recurrent1)
    dense2 = tf.keras.layers.Dense(1, activation='linear')(dense1)
    recurrent2 = tf.keras.layers.GRU(1, activation='linear', recurrent_activation='linear', return_sequences=True)(dense2)
    dense3 = tf.keras.layers.Dense(3*grid_l, activation='relu')(recurrent2)
    dense4 = tf.keras.layers.Dense(grid_l, activation='linear')(dense3)
    merge1 = tf.keras.layers.Concatenate(axis=-1)([dense2,dense4])
    dense5 = tf.keras.layers.Dense(grid_l, activation='relu')(merge1)
    merge2 = tf.keras.layers.Concatenate(axis=-1)([recurrent1,dense5])
    recurrent3 = tf.keras.layers.GRU(grid_l, activation='linear', recurrent_activation='relu', return_sequences=False, reset_after=False)(merge2)

    # Convert the GRU output to a single number and then to a word using lookup layers   
    words_3x3, numbers_3x3 = make_vocab(file='..\\notes\\words_3x3.txt')
    dense6 = tf.keras.layers.Dense(1, activation='linear')(recurrent3)
    lookup1 = tf.keras.layers.IntegerLookup(vocabulary=numbers_3x3)(dense6)
    lookup2 = tf.keras.layers.StringLookup(vocabulary=words_3x3, invert=True)(lookup1)

    full_model = tf.keras.Model(inputs=inputs, outputs=lookup2, name='full_model')
    full_model.compile(loss='mean_squared_error', metrics=['accuracy'])

    # Use layers from GRU model up to recurrent3
    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1, 0],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1, 0],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1, 0],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1, 0],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1, 0],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1, 0],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1, 0],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1, 0]])
    w1 = w0.reshape(3,3,1,11)
    b0 = np.array([-2.5, -3.5, -4.5, -6.5, -7.5, -4.5, -6.5, -4.5, -6.5, -4.5, 0])
    full_model.layers[2].set_weights([2*w1, 2*b0])

    w2 = np.arange(1,12).reshape(11,1)
    w3 = np.zeros((1,1))
    b1 = np.zeros(1)
    full_model.layers[5].set_weights([w2, w3, b1])

    w4 = np.ones((1,2))
    b2 = np.array([0,-1])
    full_model.layers[6].set_weights([w4, b2])

    w5 = np.array([[1],[-1]])
    b3 = np.zeros(1)
    full_model.layers[7].set_weights([w5, b3])

    w6 = np.array([[-1,0,0]])
    w7 = np.array([[0,0,1]])
    b4 = np.array([[1,1,1],[0,0,0]])
    full_model.layers[8].set_weights([w6, w7, b4])

    w8 = np.ones((1,3*grid_l))
    b5 = np.repeat(-np.arange(grid_l),3)
    b5[1::3] = -np.arange(grid_l)-1
    b5[2::3] = -np.arange(grid_l)-2
    full_model.layers[9].set_weights([w8, b5])

    w9 = np.zeros((3*grid_l,grid_l))
    for j in range(grid_l):
        w9[(j*3):((j+1)*3),j] = np.array([1,-2,1])
    b6 = np.zeros(grid_l)
    full_model.layers[10].set_weights([w9, b6])

    w10 = np.zeros((1+grid_l,grid_l))
    w10[0,:] = np.ones(grid_l)
    w10[1:(1+grid_l),:] = np.eye(grid_l)
    b7 = -np.ones(grid_l)
    full_model.layers[12].set_weights([w10, b7])

    w11 = np.zeros((1+grid_l, 3*grid_l))
    w11[0,(2*grid_l):(3*grid_l)] = np.ones(grid_l)
    w11[1:(1+grid_l),0:grid_l] = -np.eye(grid_l)
    w12 = np.zeros((grid_l,3*grid_l))
    b8 = np.zeros(3*grid_l)
    b8[0:grid_l] = np.ones(grid_l)
    full_model.layers[14].set_weights([w11, w12, b8])

    w13 = np.array([11]*grid_l)**np.arange(grid_l)
    w13 = w13.reshape(grid_l, 1)
    b9 = np.zeros(1)
    full_model.layers[15].set_weights([w13, b9])



zero_grid = np.zeros([6,20])
input_grid = write_letter(zero_grid,"y",0,0)
input_grid = write_letter(input_grid,"o",1,4)
input_grid = write_letter(input_grid,"u",2,8)
input_grid = write_letter(input_grid,"t",0,13)
input_grid = write_letter(input_grid,"h",3,17)


full_model(p=6,q=20)

full_model.predict(input_grid.reshape(1,6,20),verbose=0)

y = full_model.predict(input_grid.reshape(1,6,20),verbose=0)
[z.decode() for z in y.reshape(1,)]




# Try to implement again with a grid that may contain multiple words
def pq_conv_model(p, q, w):
    global pq_conv_model
    grid_p = int(np.floor(p/w))
    grid_q = int(np.floor(q/3))
    inputs = tf.keras.Input(shape=(p, q))
    reshape1 = tf.keras.layers.Reshape((p, q, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(11, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(grid_p,3), strides=(grid_p,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Permute((1,3,2))(pooling)
    sum = tf.keras.layers.Dense(1, activation='linear')(reshape2)
    pq_conv_model = tf.keras.Model(inputs=inputs, outputs=sum, name='pq_conv_model')
    pq_conv_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

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

    pq_conv_model.layers[2].set_weights([2*w1, 2*b0])

    w2 = np.ones([grid_p,1])

    b1 = np.zeros(1)

    pq_conv_model.layers[5].set_weights([w2, b1])


zero_grid = np.zeros([12,18])
input_grid = write_letter(zero_grid,"c",0,0)
input_grid = write_letter(input_grid,"i",1,4)
input_grid = write_letter(input_grid,"t",2,8)
input_grid = write_letter(input_grid,"y",3,12)

input_grid = write_letter(zero_grid,"j",6,0)
input_grid = write_letter(input_grid,"u",5,4)
input_grid = write_letter(input_grid,"l",7,8)
input_grid = write_letter(input_grid,"y",9,12)

   

pq_conv_model(p=12,q=18,w=2)

np.shape(pq_conv_model.predict(input_grid.reshape(1,12,18),verbose=0)) #(1,1,q/3,1)
#np.shape(pq_conv_model.layers[5].get_weights()[0]) # (11,1)
#np.shape(pq_conv_model.layers[5].get_weights()[1]) # (1,)
pq_conv_model.predict(input_grid.reshape(1,12,18),verbose=0)

# currently writes count for each letter per word separately