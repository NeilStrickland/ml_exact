"""Small Keras self-play experiment compatible with the players in ox.py.

The network has nine *action values*, not move probabilities. Every state
given to it has channel 0 = player to move, channel 1 = the opponent.
One shared network plays both sides. A nonterminal target is
    -max(Q_target(next_state, legal_next_action)).
No minimax labels or hand-written strategy are used during training.

Example (put this file next to ox.py)::

    from ox import OXGame, RandomPlayer
    from ox_selfplay import SelfPlayTrainer, MinimaxPlayer, evaluate

    trainer = SelfPlayTrainer(hidden=(64, 64), seed=7)
    trainer.train(games=30000)
    print(evaluate(trainer.model))
    player = trainer.player()
    print(OXGame(player, MinimaxPlayer()).ascii())
    print(OXGame(RandomPlayer(), player).ascii())
    trainer.model.save("ox_q.keras")

    # Later, for inference:
    import keras
    from ox_selfplay import QPlayer
    player = QPlayer(keras.models.load_model("ox_q.keras", compile=False))

Requires NumPy and Keras 3 with a training backend already configured.
Select KERAS_BACKEND before importing Keras if using a non-default backend.
The default architecture has 5961 parameters; hidden=(20, 20) has 989.
Training budgets/hyperparameters are starting points, not a guarantee of
perfect play. Evaluate independently; try several random seeds.

References:
https://keras.io/examples/rl/deep_q_network_breakout/
Sutton and Barto, Reinforcement Learning, sections on tic-tac-toe and Q-learning.
"""

from functools import lru_cache
import numpy as np


WIN_LINES = np.array([
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 4, 8], [2, 4, 6],
])

# A permutation gives the OLD cell occupying each NEW cell. Its inverse
# sends the old action index to the corresponding new action index.
_grid = np.arange(9).reshape(3, 3)
SYMMETRIES = np.array([
    np.rot90(g, k).ravel()
    for g in (_grid, np.fliplr(_grid)) for k in range(4)
])
INVERSE_SYMMETRIES = np.argsort(SYMMETRIES, axis=1)


def legal_mask(states):
    """Accept a single (9, 2) state or a batch (..., 9, 2)."""
    return np.asarray(states).sum(axis=-1) == 0


def has_win(state, player=0):
    return bool(np.any(np.asarray(state)[WIN_LINES, player].sum(axis=1) == 3))


def relative_state(board, player=None):
    """Convert an ox.OXBoard to an independent player-relative array."""
    if player is None:
        player = int(board.state.sum()) % 2  # O goes first in ox.py.
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    return np.asarray(board.state[:, [player, 1 - player]], dtype=np.float32).copy()


def step(state, action):
    """Play one legal move; the returned state is the opponent's viewpoint.

    Rewards refer to the player who just moved. Terminal winning moves get
    +1 and terminal draws get 0. Losses propagate through the negative backup.
    Both the input and returned observation are independent arrays.
    """
    if has_win(state, 0) or has_win(state, 1):
        raise ValueError("cannot play after a win")
    if not isinstance(action, (int, np.integer)) or not 0 <= action < 9:
        raise ValueError("action must be an integer from 0 to 8")
    if not legal_mask(state)[action]:
        raise ValueError("occupied square")
    after = np.array(state, dtype=np.float32, copy=True)
    after[action, 0] = 1
    win = has_win(after)
    done = win or not legal_mask(after).any()
    return after[:, ::-1].copy(), float(win), bool(done)


def _predict(model, states):
    return np.asarray(model.predict_on_batch(np.asarray(states, dtype=np.float32)))


def negamax_targets(target_model, next_states, rewards, done, gamma=1.0):
    """Detached targets; never maximize over an empty terminal action set."""
    targets = np.array(rewards, dtype=np.float32, copy=True)
    live = np.flatnonzero(~np.asarray(done, dtype=bool))
    if live.size:
        following = np.asarray(next_states)[live]
        legal = legal_mask(following)
        if not legal.any(axis=1).all():
            raise ValueError("nonterminal successor has no legal action")
        # Predict the full batch to keep the compiled backend's batch shape
        # fixed, but use only live rows in the maximum and target arithmetic.
        q_next = _predict(target_model, next_states)[live]
        targets[live] -= gamma * np.where(legal, q_next, -np.inf).max(axis=1)
    return targets


def make_q_model(hidden=(64, 64)):
    """Use ordinary serializable Keras layers; leave output unnormalized."""
    import keras

    state = keras.Input(shape=(9, 2), name="state")
    x = keras.layers.Flatten()(state)
    for width in hidden:
        x = keras.layers.Dense(width, activation="relu")(x)
    q = keras.layers.Dense(9, activation="linear", name="q_values")(x)
    return keras.Model(state, q, name="ox_q")


class QPlayer:
    """Duck-types ox.OXPlayer; can be passed directly to ox.OXGame.

    Evaluation is greedy by default. Exact ties go to the first legal index,
    the same rule used by evaluate(). Training breaks ties randomly instead.
    """

    def __init__(self, model, epsilon=0.0, seed=None):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0, 1]")
        self.model = model
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def play(self, board, player=None):
        state = relative_state(board, player)
        legal = np.flatnonzero(legal_mask(state))
        if not legal.size or has_win(state, 0) or has_win(state, 1):
            return None
        if self.epsilon and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        q = _predict(self.model, state[None])[0]
        return int(legal[np.argmax(q[legal])])


class SelfPlayTrainer:
    """One-ply negamax DQN with replay, a target copy, and board symmetries.

    Only the chosen action contributes to the Huber loss. A second Keras
    model selects that output using a one-hot action input; it shares all
    network weights with self.model. This avoids a backend-specific custom
    gradient loop or a custom loss needed when loading a saved Q-network.

    Saving self.model saves a playing network, not the replay buffer, target
    network, random generator state, or complete training checkpoint.
    """

    def __init__(self, hidden=(64, 64), learning_rate=1e-3, seed=7,
                 capacity=50000, batch_size=128, warmup=512,
                 target_interval=200, gamma=1.0, augment=True):
        import keras

        if not 1 <= batch_size <= warmup <= capacity:
            raise ValueError("require 1 <= batch_size <= warmup <= capacity")
        if target_interval < 1 or not 0 <= gamma <= 1:
            raise ValueError("invalid target_interval or gamma")
        keras.utils.set_random_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.model = make_q_model(hidden)
        self.target = keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())

        chosen_action = keras.Input(shape=(9,), name="chosen_action")
        chosen_q = keras.layers.Dot(axes=1)([self.model.output, chosen_action])
        self.training_model = keras.Model(
            [self.model.input, chosen_action], chosen_q)
        self.training_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss=keras.losses.Huber(),
        )

        self.capacity = capacity
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_interval = target_interval
        self.gamma = gamma
        self.augment = augment
        self.states = np.empty((capacity, 9, 2), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=bool)
        self.size = self.cursor = self.updates = self.games = 0
        self.history = []

    def _remember(self, state, action, reward, following, done):
        i = self.cursor
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = following
        self.done[i] = done
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self):
        ids = self.rng.integers(self.size, size=self.batch_size)
        states = self.states[ids]
        following = self.next_states[ids]
        actions = self.actions[ids]
        if self.augment:
            syms = self.rng.integers(8, size=self.batch_size)
            perm = SYMMETRIES[syms]
            rows = np.arange(self.batch_size)[:, None]
            states = states[rows, perm]
            following = following[rows, perm]
            actions = INVERSE_SYMMETRIES[syms, actions]
        y = negamax_targets(self.target, following, self.rewards[ids],
                            self.done[ids], self.gamma)
        one_hot = np.eye(9, dtype=np.float32)[actions]
        # reset_metrics makes the returned loss describe this update.
        self.training_model.reset_metrics()
        loss = float(self.training_model.train_on_batch(
            [states, one_hot], y[:, None]))
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target.set_weights(self.model.get_weights())
        return loss

    def train(self, games=30000, epsilon_start=1.0, epsilon_end=0.1,
              decay_games=20000, updates_per_game=2, report_every=1000):
        """Play additional games; returns the newly recorded progress rows.

        Epsilon decays linearly according to cumulative self.games, so calling
        train() again continues the same schedule. Both sides explore. The
        printed game outcomes are diagnostics, not measures of playing strength.
        Use evaluate(self.model) to assess the greedy policy independently.
        """
        if games < 0 or decay_games < 1 or updates_per_game < 1:
            raise ValueError("invalid training budget")
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
        new_history = []
        counts = np.zeros(3, dtype=int)  # O wins, X wins, draws
        last_loss = None
        for local_game in range(games):
            fraction = min(self.games / decay_games, 1.0)
            epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
            state = np.zeros((9, 2), dtype=np.float32)
            for ply in range(9):
                legal = np.flatnonzero(legal_mask(state))
                if self.rng.random() < epsilon:
                    action = int(self.rng.choice(legal))
                else:
                    q = _predict(self.model, state[None])[0]
                    best = legal[q[legal] == q[legal].max()]
                    action = int(self.rng.choice(best))
                following, reward, done = step(state, action)
                self._remember(state, action, reward, following, done)
                if done:
                    counts[ply % 2 if reward else 2] += 1
                    break
                state = following
            self.games += 1
            if self.size >= self.warmup:
                for _ in range(updates_per_game):
                    last_loss = self._update()
            if report_every and (self.games % report_every == 0 or local_game == games - 1):
                row = dict(games=self.games, epsilon=float(epsilon), loss=last_loss,
                           o_wins=int(counts[0]), x_wins=int(counts[1]),
                           draws=int(counts[2]))
                self.history.append(row)
                new_history.append(row)
                print(row, flush=True)
                counts[:] = 0
        return new_history

    def player(self, epsilon=0.0, seed=None):
        return QPlayer(self.model, epsilon=epsilon, seed=seed)


# Everything below is evaluation only. Training never calls these routines.
def _key(state):
    return tuple((np.asarray(state)[:, 0] - np.asarray(state)[:, 1]).astype(int))


def _state(key):
    a = np.asarray(key)
    return np.stack([a == 1, a == -1], axis=1).astype(np.float32)


def _won(key, player):
    return any(all(key[i] == player for i in line) for line in WIN_LINES)


def _successor(key, action):
    following = list(key)
    following[action] = 1
    return tuple(-x for x in following)


@lru_cache(maxsize=None)
def minimax_value(key):
    """Exact value for the current player: -1 (loss), 0 (draw), +1 (win)."""
    if _won(key, 1):
        return 1
    if _won(key, -1):
        return -1
    legal = [i for i, x in enumerate(key) if x == 0]
    if not legal:
        return 0
    return max(-minimax_value(_successor(key, a)) for a in legal)


def exact_action_values(state):
    """Nine exact values; illegal or post-terminal actions are -infinity."""
    key = _key(state)
    q = np.full(9, -np.inf)
    if _won(key, 1) or _won(key, -1):
        return q
    for a, x in enumerate(key):
        if x == 0:
            q[a] = -minimax_value(_successor(key, a))
    return q


class MinimaxPlayer:
    """Exact reference player, with the same play() interface as ox.py."""

    def play(self, board, player=None):
        q = exact_action_values(relative_state(board, player))
        return int(q.argmax()) if np.isfinite(q).any() else None


@lru_cache(maxsize=1)
def _reachable_keys():
    seen = set()
    live = []

    def visit(key):
        if key in seen:
            return
        seen.add(key)
        if _won(key, 1) or _won(key, -1) or 0 not in key:
            return
        live.append(key)
        for a, x in enumerate(key):
            if x == 0:
                visit(_successor(key, a))

    visit((0,) * 9)
    return tuple(live)


def evaluate(model, batch_size=256):
    """Exhaustively test the deterministic, masked-greedy playing policy.

    optimal_move_fraction accepts every equally good minimax move.
    worst_case_as_o/x are exact best-response results from the empty board:
    -1 means some opponent can force a loss; 0 means the policy guarantees
    at least a draw. A draw against one fixed minimax tie-breaker is weaker.
    No enumeration, solver results, or evaluation data enter the training loop.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    keys = _reachable_keys()
    states = np.array([_state(key) for key in keys])
    predictions = np.concatenate([
        _predict(model, states[i:i + batch_size])
        for i in range(0, len(states), batch_size)
    ])
    if not np.isfinite(predictions).all():
        raise ValueError("network predictions contain NaN or infinity")
    actions = np.where(legal_mask(states), predictions, -np.inf).argmax(axis=1)
    policy = dict(zip(keys, map(int, actions)))
    optimal = avoidable_losses = missed_wins = 0
    for key, action in policy.items():
        best = minimax_value(key)
        chosen = -minimax_value(_successor(key, action))
        optimal += chosen == best
        avoidable_losses += best >= 0 and chosen == -1
        missed_wins += best == 1 and chosen < 1

    def worst_case(as_player):
        @lru_cache(maxsize=None)
        def visit(key):
            ply = sum(x != 0 for x in key)
            turn = ply % 2  # O=0 goes first; X=1 goes second.
            sign = 1 if turn == as_player else -1
            if _won(key, 1):
                return sign
            if _won(key, -1):
                return -sign
            if 0 not in key:
                return 0
            if turn == as_player:
                return visit(_successor(key, policy[key]))
            return min(visit(_successor(key, a))
                       for a, x in enumerate(key) if x == 0)

        return visit((0,) * 9)

    return dict(states=len(keys), optimal_moves=int(optimal),
                optimal_move_fraction=float(optimal / len(keys)),
                avoidable_losses=int(avoidable_losses), missed_wins=int(missed_wins),
                worst_case_as_o=int(worst_case(0)), worst_case_as_x=int(worst_case(1)))
