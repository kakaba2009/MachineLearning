from __future__ import print_function

import os
import time
import math
import warnings
import numpy as np
import tensorflow as tf
import src.mylib.mcalc as mcalc
import src.mylib.mlstm as mlstm
import src.mylib.mutils as utl
from tensorflow.contrib import rnn
from sklearn.model_selection import KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

tf.set_random_seed(0)  # fix random seed

loaded = False
modelSaved = "../model/JPYRMSPropLinear12x6xD6.h5"

SEQLEN = 5
BATCHSIZE = 10
output_dim = 5
CELLSIZE = 16
NLAYERS = 1
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 1.0    # no dropout
epochs_num = 1

ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 101)
ds = ds[['Close']]

#T = mcalc.vector_delay_embed(ds, output_dim, 1)


def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch

def mshape(x):
    # reshape input to be [samples, time steps, features]
    return np.reshape(x, (x.shape[0],  -1, x.shape[1]))

kf = KFold(n_splits=3, shuffle=False, random_state=None)

lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.float32, [None, SEQLEN], name='X')    # [ BATCHSIZE, SEQLEN ]
Xo = X
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.float32, [None, SEQLEN], name='Y_')  # [ BATCHSIZE, SEQLEN ]
Yo_ = Y_
# input state
Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

# using a NLAYERS=1 layers of GRU cells, unrolled SEQLEN=5 times
# dynamic_rnn infers SEQLEN from the size of the inputs Xo

onecell = rnn.GRUCell(CELLSIZE)
dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
multicell = rnn.MultiRNNCell([dropcell]*NLAYERS, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

H = tf.identity(H, name='H')  # just to give it a name

# Softmax layer implementation:
# Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]
# then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
# From the readout point of view, a value coming from a cell or a minibatch is the same thing

Yflat = tf.reshape(Yr, [-1, CELLSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Ylogits = tf.contrib.layers.fully_connected(Yflat, output_dim)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, output_dim])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.losses.mean_squared_error(predictions=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
Yo = Ylogits   # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.reshape(Yo, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Define loss and optimizer
#cost = tf.losses.mean_squared_error(labels=y, predictions=prediction)
#tf.summary.scalar("cost", cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# stats for display
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.metrics.mean_squared_error(Y_, Y))
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
for x, y_, epoch in rnn_minibatch_sequencer(ds.values, BATCHSIZE, SEQLEN, nb_epochs=epochs_num):
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

    # display progress bar
    print(step)

    # loop state around
    istate = ostate
    step += BATCHSIZE * SEQLEN
