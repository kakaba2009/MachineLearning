from __future__ import print_function

import os
import time
import math
import warnings
import numpy as np
import tensorflow as tf
import src.mylib.mlstm as mlstm
import src.mylib.mutils as utl
from tensorflow.contrib import rnn
from sklearn.model_selection import KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

tf.set_random_seed(0)  # fix random seed

loaded = False
modelSaved = "../model/JPYRMSPropLinear12x6xD6.h5"

SEQLEN = 2
BATCHSIZE = 10
output_dim = 1
CELLSIZE = 16
NLAYERS = 1
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 1.0    # no dropout
epochs_num = 10

ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 21)
ds = ds[['Close']]
ds = ds.values
ds = np.reshape(ds, -1)

def mshape(x):
    # reshape input to be [samples, time steps, features]
    return np.reshape(x, (x.shape[0],  -1, x.shape[1]))

kf = KFold(n_splits=3, shuffle=False, random_state=None)

lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.float32, [None, SEQLEN], name='X')    # [ BATCHSIZE, SEQLEN ]
Xo = tf.reshape(X, [-1, SEQLEN, output_dim])    # [ BATCHSIZE, SEQLEN, output_dim ]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.float32, [None, SEQLEN], name='Y_')  # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.reshape(Y_, [-1, SEQLEN, output_dim])  # [ BATCHSIZE, SEQLEN, output_dim ]
# input state
Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS], name='Hin')  # [ BATCHSIZE, CELLSIZE * NLAYERS]

# using a NLAYERS=1 layers of GRU cells, unrolled SEQLEN=5 times
# dynamic_rnn infers SEQLEN from the size of the inputs X

onecell = rnn.GRUCell(CELLSIZE)
dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
multicell = rnn.MultiRNNCell([dropcell]*NLAYERS, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin, time_major=False)
# Yr: [ BATCHSIZE, SEQLEN, CELLSIZE ]
# H:  [ BATCHSIZE, CELLSIZE*NLAYERS ] # this is the last state in the sequence

H = tf.identity(H, name='H')  # just to give it a name

# Softmax layer implementation:
# Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, output_dim ] => [ BATCHSIZE x SEQLEN, output_dim ]
# then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
# From the readout point of view, a value coming from a cell or a minibatch is the same thing

Yflat = tf.reshape(Yr, [-1, CELLSIZE])    # [ BATCHSIZE x SEQLEN, CELLSIZE ]
Ylogits = tf.contrib.layers.fully_connected(Yflat, output_dim)     # [ BATCHSIZE x SEQLEN, output_dim ]
Yflat_ = tf.reshape(Y_, [-1, output_dim])     # [ BATCHSIZE x SEQLEN, output_dim ]
loss = tf.losses.mean_squared_error(predictions=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
Y = Ylogits   # [ BATCHSIZE x SEQLEN, output_dim ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Define loss and optimizer
#cost = tf.losses.mean_squared_error(labels=y, predictions=prediction)
#tf.summary.scalar("cost", cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# stats for display
seqloss = tf.reduce_mean(loss)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(loss)
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
# you can compare training and validation curves visually in Tensorboard.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("../log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("../log/" + timestamp + "-validation")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1)

# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN

# init
istate = np.zeros([BATCHSIZE, CELLSIZE * NLAYERS])  # initial zero input state
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

# training loop
for x, y_, epoch in utl.rnn_minibatch_sequencer(ds, BATCHSIZE, SEQLEN, nb_epochs=epochs_num):
    # train on one minibatch
    feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
    _, y, ostate, smm = sess.run([train_step, Y, H, summaries], feed_dict=feed_dict)

    # save training data for Tensorboard
    summary_writer.add_summary(smm, step)

    # display a visual validation of progress (every 50 batches)
    if step % _50_BATCHES == 0:
        feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCHSIZE}  # no dropout for validation
        y, l, bl, acc = sess.run([Y, seqloss, batchloss, accuracy], feed_dict=feed_dict)

    # run a validation step every 50 batches
    # The validation text should be a single sequence but that's too slow (1s per 1024 chars!),
    # so we cut it up and batch the pieces (slightly inaccurate)
    # tested: validating with 5K sequences instead of 1K is only slightly more accurate, but a lot slower.
    if step % _50_BATCHES == 0:
        ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        utl.print_validation_stats(ls, acc)
        # save validation data for Tensorboard
        validation_writer.add_summary(smm, step)

    # save a checkpoint (every 500 batches)
    if step // 10 % _50_BATCHES == 0:
        saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)

    print("x", x)
    print("y_", y_)

    # loop state around
    istate = ostate
    step += BATCHSIZE * SEQLEN
