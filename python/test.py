
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

src_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.realpath(os.path.join(src_dir, '../models'))

exact_model = None
approx_model = None

# two English words we want to translate to German
zero = "zero"
one  = "one "

# convert to list of numbers
in_zero = [ord(x) for x in list(zero)] #[122,101,114,111]
in_one  = [ord(x) for x in list(one)]  #[111,110,101,32]


# the desired outputs
out_null = [ord(x) for x in list("null")] #[110,117,108,108]
out_eins = [ord(x) for x in list("eins")] #[101,105,110,115]

# so we want function that does:
# [122 -> 110
#  101 -> 117
#  114 -> 108
#  111 -> 108]
# and
# [111 -> 101
#  110 -> 105
#  101 -> 110
#  32  -> 115]

# manually solve 4 sets of simultaneous equations
# w0 = [[9/11, 0,     0,     0    ],
#       [0,    -12/9, 0,     0    ],
#       [0,    0,     -2/13, 0    ],
#       [0,    0,     0,     -7/79]]

# b0 = [110-(122*9)/11,
#       105+(110*12)/9,
#       108+(114*2)/13,
#       108+(111*7)/79]

eng_in = [in_zero, in_one]
deu_out = [out_null, out_eins]


def make_exact_model():
    global exact_model
    exact_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, input_dim=4, activation="linear")
    ])

    w = np.array([[9/11, 0, 0, 0],
                  [0, -12/9, 0, 0],
                  [0, 0, -2/13, 0],
                  [0, 0, 0, -7/79]])

    b = np.array([110-(122*9)/11,
                  105+(110*12)/9,
                  108+(114*2)/13,
                  108+(111*7)/79])

    exact_model.layers[0].set_weights([w,b])

    exact_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

    exact_model.evaluate(eng_in, deu_out)

make_exact_model()

in_zero = [ord(x) for x in list("zero")]
in_one  = [ord(x) for x in list(one)]
print(exact_model.predict([in_zero]))
print(exact_model.predict([in_one]))
