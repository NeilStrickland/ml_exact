# Here we build a model to play noughts and crosses
# Eventually we hope to do this by Reinforcement Learning, but
# code for that does not work properly yet.  Instead we have
# coded the usual algorithm and trained a model to mimic that.

# We effectively treat the board as a linear array of 9 cells.
# Each game position is represented as an array of shape (9, 2)
# with a 1 in position (i, 0) if there is an O in position i,
# and a 1 in position (i, 1) if there is an X in position i.
# Player 0 plays Os and player 1 plays Xs.

from re import I
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import os
from itertools import combinations

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

src_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.realpath(os.path.join(src_dir,'../models'))

model = None

BATCH_SIZE = 64
SHUFFLE_SIZE = 100
EPOCHS = 500

board = range(9)

# The array below encodes the rules which determine when a 
# player has won.  For example, wins[0] has 1s in positions 
# 0, 3 and 6, indicating that a player has won if they 
# have counters in positions 0, 3 and 6.
wins = np.array([
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 1, 0, 0]
])

# ee[i] is the vector of length 9 with a single 1 in position i
# eee[i, j] is the array of shape (9, 2) with a single 1 in position (i, j)
ee = np.eye(9)
eee = np.zeros([9, 2, 9, 2])


def make_eee():
    global eee
    for i in range(9):
        for j in range(2):
            eee[i, j, i, j] = 1


make_eee()

# corners[i] is 1 iff position i is a corner of the board
# edges[i] is 1 iff position i is the middle of one edge of the board
corners = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1])
edges   = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])

# opposite[i] is the position opposite position i
opposite = [8, 7, 6, 5, 4, 3, 2, 1, 0]

empty_board = np.zeros([9, 2])


def add_o(pq, i): pq[i, 0] = 1


def add_x(pq, i): pq[i, 1] = 1

# has_win(pq, 0) is True if, in the game state described by the 
# array pq, player 0 has filled a line (and so has won, if the
# game state was reached by legal play).  has_win(pq, 1) is similar.
# has_win(pq) = has_win(pq, None) = has_win(pq, 0) or has_win(pq, 1)
def has_win(pq, player=None):
    w = [np.max(np.matmul(wins, pq[:, j])) >= 3 for j in [0, 1]]
    if player is None:
        return w[0] or w[1]
    else:
        return w[player]

# plays(pq) is the list of positions which are empty (in the game 
# state described by the array pq), and so are available to the 
# next player.
def plays(pq, player=0):
    n = []
    for i in range(9):
        if pq[i, 0] == 0 and pq[i, 1] == 0:
            n.append(i)
    return n


# Use manual NN to replicate plays()
def plays_model():
    global plays_model
    inputs = tf.keras.Input(shape=(9, 2))
    reshape = tf.keras.layers.Reshape((18,))(inputs)
    outputs = tf.keras.layers.Dense(9, activation='linear')(reshape)
    plays_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="plays_model")

    w0 = np.zeros([18,9])
    w0[(0,2,4,6,8,10,12,14,16),:] -= np.eye(9) # player 0 moves
    w0[(1,3,5,7,9,11,13,15,17),:] -= np.eye(9) # player 1 moves

    b0 = np.ones(9) # adjust from {-1, 0} to {0, 1}

    plays_model.layers[2].set_weights([w0, b0])
    return plays_model

plays_model()
pq = empty_board.copy()
plays_model.predict(pq.reshape(1, 9, 2))
add_o(pq, 1)
plays_model.predict(pq.reshape(1, 9, 2))
add_x(pq, 4)
plays_model.predict(pq.reshape(1, 9, 2))


# near_wins(pq, i) is the list of positions at which player i could 
# play and thereby immediately win.
def near_wins(pq, player=0):
    n = []
    for i in range(9):
        if pq[i, 0] == 0 and pq[i, 1] == 0 and has_win(pq + eee[i, player], player):
            n.append(i)
    return n


# forks(pq, i) is the list of positions at which player i could 
# play and thereby create two near-wins.
def forks(pq, player=0):
    f = []
    for i in range(9):
        if pq[i, player] == 0 and pq[i, 1-player] == 0 and \
                len(near_wins(pq + eee[i, player], player)) > 1:
            f.append(i)
    return f


# swap(pq) represents the same game state as pq but with the Os and Xs exchanged
def swap(pq):
    return np.array([[pq[i, 1-j] for j in range(2)] for i in range(9)])


# This function expects a list moves = [moves_o, moves_x], where 
# moves_o and moves_x are disjoint lists of distinct board positions.
# It returns the corresponding game state pq as an array of shape (9, 2)
def board_from_moves(moves):
    s = empty_board.copy()
    for i in [0,1]:
        for j in moves[i]:
            s += eee[j, i]
    return s


# random_move(pq, i) returns a randomly selected legal move for
# player i in game state pq.  The return value is just an integer
# (or None if the board is already full).
def random_move(pq, player=0):
    if has_win(pq) or len(plays(pq)) == 0:
        return None
    return np.random.choice(plays(pq, player))


# suggest_move(pq, i) returns a suggested move for player i in 
# game state pq.  The suggestions are optimal in the sense that 
# a win or draw is guaranteed if the suggestions are followed
# throughout the game.  The function returns None if the board 
# is already full.
def suggest_move(pq, player):
    if player == 0:
        pq0 = pq.copy()
        qp0 = swap(pq0)
    else:
        qp0 = pq.copy()
        pq0 = swap(qp0)
    if has_win(pq0) or len(plays(pq0)) == 0:
        return None
    n = near_wins(pq0)
    if len(n) > 0: return n[0] # win if poss
    m = near_wins(qp0)
    if len(m) > 0: return m[0] # not lose if poss
    f = forks(pq0)
    if len(f) > 0: return f[0] # take fork if poss
    g = forks(qp0)
    if len(g) == 1: return g[0] # prevent fork if poss
    for i in g:
        pq1 = pq0 + eee[i, 0]
        qp1 = swap(pq1)
        if len(near_wins(pq1)) > 0 and len(forks(qp1)) > 0:
            return i # play somewhere that creates a near win and doesnt give opportunity to fork
    # edit: if going first can start in corner instead of middle
    if sum(sum(pq0)) == 0: return 0
    if pq0[4, 0] == 0 and pq0[4, 1] == 0: return 4 # play middle if poss
    for i in [0, 2, 6, 8]:
        if pq0[i, 0] == 0 and pq0[i, 1] == 0 and pq0[8-i, 1] == 1: # play opposite corner to opponent
            return i
    for i in [0, 2, 6, 8, 1, 3, 5, 7]: # otherwise prioritise corners -> sides
        if pq0[i, 0] == 0 and pq0[i, 1] == 0:
            return i
    return None


# this can be easily beaten going first by [0, 8, 6, 7]


# This function displays the game state pq and prompts the user
# to enter a move for the specified player.
def input_move(pq, player=0):
    if player == 0:
        s = 'O'
    else:
        s = 'X'
    show_board_ascii(pq)
    prompt = 'Enter position (0-8) for next ' + s + ': '
    i = input(prompt)
    try:
        i = int(i)
    except ValueError:
        return None
    if i < 0 or i > 8:
        i = None
    return i


# This function sets various global variables.
# all_states is the list of all game states that could be seen 
# by player 0 in legal play (with either player 0 or player 1 
# starting the game).  It only includes the 4519 states in which 
# player 0 is required to respond; states are excluded if either 
# player has already won.  
#
# all_states_suggestions is the corresponding list of suggested
# moves, with one-hot encoding.  In other words, if the suggestion
# for state all_states[i] is to play in position j, then 
# all_states_suggestions[i] is the basis vector ee[j]
#
# all_dataset packages all_states and all_states_suggestions in
# a tensorflow dataset (shuffled and batched)
def make_all_states():
    global states, all_states, all_states_suggestions, all_dataset
    all_states = []
    states = [[empty_board.copy()]]
    for i in range(1,9):
        states.append([])
        for s in states[i-1]:
            p = plays(s)
            k = 8
            while s[k, i % 2] == 0 and k >= 0: 
                k = k-1
            for j in p:
                s1 = s + eee[j, i % 2]
                if j > k and not has_win(s1):
                    states[i].append(s1)
        all_states += states[i]
    
    all_states_suggestions = list(map(
        lambda pq: ee[suggest_move(pq, 1)], 
        all_states
    ))

    all_dataset = tf.data.Dataset.from_tensor_slices((all_states, all_states_suggestions))
    all_dataset = all_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)

    return all_states


make_all_states()

# HL: to access first pair of (state, suggestion) use
# all_dataset = tf.data.Dataset.from_tensor_slices((all_states, all_states_suggestions))
# list(all_dataset.as_numpy_iterator())[0]


# edit: remove Q model


######################################################################
# This section attempts to train a model to play noughts and 
# crosses.  The model takes a game state as input and produces
# a probability vector of length 9 as output.

def make_model(p=20, q=20, r=0.001):
    global model
    inputs = tf.keras.Input(shape=(9, 2))
    reshape = tf.keras.layers.Reshape((18,))(inputs)
    hidden0 = tf.keras.layers.Dense(p, activation='relu')(reshape)
    hidden1 = tf.keras.layers.Dense(q, activation='relu')(hidden0)
    outputs = tf.keras.layers.Dense(9, activation='softmax')(hidden1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model")
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def save_model():
    l = model.layers
    p = l[3].input_shape[1]
    q = l[4].input_shape[1]
    model_dir = os.path.join(models_dir, 'ox_model_' + repr(p) + '_' + repr(q))
    model.save(model_dir)


def load_model(p=20, q=20):
    model_dir = os.path.join(models_dir, 'ox_model_' + repr(p) + '_' + repr(q))
    model = tf.keras.models.load_model(model_dir)
    return model






# This function applies the model to a game state pq to generate
# a probability distribution on the board positions, and then 
# selects a position randomly in accordance with that distribution.
# (If the model is not well-trained then the move may be illegal, 
# resulting in an immediate loss.)
def model_move(pq, player=0):
    if player == 0:
        pq0 = pq.copy()
    else:
        pq0 = swap(pq.copy())
    prob = model.predict(pq0.reshape(1, 9, 2), verbose=0)[0]
    choices = range(9)
    #return np.random.choice(choices, p=prob)
    return choices[np.argmax(prob)]


# This function displays the given game state as an image 
# using matplotlib
def show_board(pq, winner=None):
    col_o = 'red'
    col_x = 'blue'
    if winner == 0: col_x = 'lightblue'
    if winner == 1: col_o = 'lightcoral'
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_axis_off()
    ax.set_xlim((0, 3))
    ax.set_ylim((0, 3))
    ax.add_line(matplotlib.lines.Line2D([1, 1], [0, 3], color='grey'))
    ax.add_line(matplotlib.lines.Line2D([2, 2], [0, 3], color='grey'))
    ax.add_line(matplotlib.lines.Line2D([0, 3], [1, 1], color='grey'))
    ax.add_line(matplotlib.lines.Line2D([0, 3], [2, 2], color='grey'))
    for i in range(3):
        for j in range(3):
            k = i+3*j
            if pq[k, 0] == 1:
                ax.add_patch(matplotlib.patches.Circle([i + 0.5, j + 0.5], 0.3, color=col_o, fill=False))
            if pq[k, 1] == 1:
                ax.add_line(matplotlib.lines.Line2D([i + 0.2, i + 0.8], [j + 0.2, j + 0.8], color=col_x))
                ax.add_line(matplotlib.lines.Line2D([i + 0.2, i + 0.8], [j + 0.8, j + 0.2], color=col_x))


# This function prints an ASCII art representation of the specified game state
def show_board_ascii(pq, winner=None):
    sym_o = 'O'
    sym_x = 'X'
    if winner == 0: sym_x = 'x'
    if winner == 1: sym_o = 'o'
    s = '---\n'
    for j in range(3):
        for i in range(3):
            k = i+3*(2-j)
            t = '#'
            if pq[k, 0] == 1: t = sym_o
            if pq[k, 1] == 1: t = sym_x
            s = s + t
        s = s + '\n'
    s = s + '---'
    print(s)


# This function plays a game.
# It assumes that player_o and player_x are functions like
# random_move, suggest_move, input_move and model_move:
# they should accept a game state and player index, and 
# return a board position.  By default play_game() returns
# the final board position and the index of the winner 
# (0 for O, 1 for X).  If the return_moves argument is true
# it also returns a list moves = [moves_o, moves_x] 
# specifying the moves taken in order.
def play_game(player_o, player_x, return_moves=False):
    pq = empty_board.copy()
    winner = None
    # edit: track illegal moves
    illegal = None
    moves_o = []
    moves_x = []
    while True:
        if len(plays(pq)) == 0:
            break
        i = player_o(pq, 0)
        moves_o.append(i)
        if i is None or pq[i, 0] == 1 or pq[i, 1] == 1:
            winner = 1
            illegal = 0
            break
        add_o(pq, i)
        if has_win(pq, 0):
            winner = 0
            break
        if has_win(pq, 1):
            winner = 1
            break
        if len(plays(pq)) == 0:
            break
        i = player_x(pq, 1)
        moves_x.append(i)
        if i is None or pq[i, 0] == 1 or pq[i, 1] == 1:
            winner = 0
            illegal = 1
            break
        add_x(pq, i)
        if has_win(pq, 0):
            winner = 0
            break
        if has_win(pq, 1):
            winner = 1
            break
    if return_moves:
        return pq, winner, illegal, [moves_o, moves_x]
    else:
        return pq, winner, illegal


# edit: remove step()

# edit: remove play game monitored


# The function legal_loss(pq, prob) expects pq to be a game state and
# prob to be a probability vector of length 9, interpreted as the 
# probabilities of playing in positions 0,..,8.  It returns the 
# probability of playing an illegal move, i.e. of playing in a 
# position that is already occupied.
def legal_loss(pq, prob):
    oo = tf.constant([[1],[1]], dtype=tf.float32)
    return tf.reshape(tf.linalg.matmul(tf.reshape(prob,[-1,1,9]),tf.linalg.matmul(pq,oo)),[-1])


def use_legal_loss(on=True):
    global model
    if on:
        model.compile(loss=legal_loss, optimizer=model.optimizer)
    else:
        model.compile(loss="categorical_crossentropy", optimizer=model.optimizer)


def set_learning_rate(r=0.01):
    global model
    tf.keras.backend.set_value(model.optimizer.learning_rate, r)


# This function trains the model so that it only plays legal moves.
# It does not attempt to ensure that the moves are sensible.
def train_model_rules():
    global model
    if model is None:
        make_model()
    use_legal_loss(True)
    model.compile(loss=legal_loss, optimizer=model.optimizer)
    st = tf.constant(all_states, dtype=tf.float32)
    model.fit(st, st, epochs=20)
    use_legal_loss(False)


# This function trains the model to replicate the behaviour of the
# suggest_move() function
def train_model_suggestions():
    global model
    if model is None:
        make_model()
    use_legal_loss(False)
    model.fit(all_dataset, epochs=EPOCHS, verbose=0)


# edit: remove reinforcement model


# Add residual connectionsto try and help the model learn legal moves only
def make_res_model(p=10, q=10, r=0.01):
    global res_model
    inputs = tf.keras.Input(shape=(9, 2))
    reshape = tf.keras.layers.Reshape((18,))(inputs)
    hidden0 = tf.keras.layers.Dense(p, activation='relu')(reshape)
    hidden1 = tf.keras.layers.Dense(q, activation='relu')(hidden0)
    outputs1 = tf.keras.layers.Dense(9, activation='linear')(hidden1)
    #outputs2 = tf.keras.layers.Dense(9, activation='linear')(reshape)
    outputs2 = tf.keras.layers.Dense(9, activation='linear', trainable=False)(reshape)
    resout = tf.keras.layers.Add()([outputs1,outputs2])
    resout = tf.keras.layers.Activation("softmax")(resout)
    res_model = tf.keras.Model(inputs=inputs, outputs=resout, name="model")
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    res_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    w1 = np.zeros((18,9))
    for i in range(9):
        w1[2*i,i] = 1
        w1[2*i+1,i] = 1
    w1 = -20*w1
    b1 = np.zeros(9)
    res_model.layers[5].set_weights([w1,b1])
    
    return res_model

make_res_model()

def train_res_model():
    global res_model
    if res_model is None:
        make_res_model()
    res_model.fit(all_dataset, epochs=EPOCHS, verbose=0)

train_res_model()

# Copy model_move() using softmax
# Hopefully softmax can work similar to argmax with the residual connections
def res_model_move(pq, player=0):
    if player == 0:
        pq0 = pq.copy()
    else:
        pq0 = swap(pq.copy())
    prob = res_model.predict(pq0.reshape(1, 9, 2), verbose=0)[0]
    choices = range(9)
    return np.random.choice(choices, p=prob)

# Seems we can easily get equal or better performance with p,q smaller


# Now try using the same structure but set the residual connections first
# and prevent them from being trained

def play_match(player_o, player_x, bestof=100):
    scores = [0, 0, 0]
    illegals = [0, 0]

    for game in range(bestof):
        pq, winner, illegal = play_game(player_o, player_x)
        if winner == 0: scores[1] += 1
        elif winner ==1: scores[2] += 1
        else: scores[0] += 1
        if illegal == 0: illegals[0] += 1
        elif illegal == 1: illegals[1] += 1
        #print("Game " + str(game+1) + "/" + str(bestof)+ " done.")

    return scores, illegals

#make_model(p=20, q=20, r=0.01)
#train_model_suggestions()
#play_match(model_move, suggest_move)
scores, illegals = play_match(res_model_move, suggest_move)


# here we can achieve 99% performance with res_model using
# fixed residual connections and p=q=10 for the rest of the network

# or letting the network train the residual connections and
# results in approx -20 to -40 weights on diagonals and not much else


def test_pq():
    scores = []
    illegals = []
    for size in range(10, 30):
        make_model(p=size, q=size, r=0.01)
        train_model_suggestions()
        score, illegal = play_match(suggest_move,model_move)
        scores.append(score)
        illegals.append(illegal)
        print("p,q=" + str(size))

    return scores, illegals

#scores, illegals = test_pq()
#print(scores)
#print(illegals)

# below p,q=20 it starts to reliably defend properly
# below p,q=15 it starts to struggle to not lose due to an illegal move

######################################################################

# make_model()
# train_model_rules()

# train_model_suggestions()

# while True:
#    rr, gg, mm, bb = play_game_monitored()
#    print([rr,mm])
#    show_board_ascii(board_from_moves(mm))

# make_Q_model()

# train_Q_once(5,5,5,0.5)

# make_model()
