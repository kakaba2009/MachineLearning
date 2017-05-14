from __future__ import print_function

import os
import time
import math
import warnings
import numpy as np
import tensorflow as tf
import src.mylib.mlstm as mlstm
import src.mylib.mcalc as mcalc
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

tf.set_random_seed(0)  # fix random seed

# Hyper Parameters
BATCHSIZE = 50
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
CELL_SIZE = 32      # rnn cell size
LR = 0.001          # learning rate

ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 1000)
ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 1000)
ds = ds[['Close']]
ds = mcalc.vector_delay_embed(ds, INPUT_SIZE, 1)

S, T = mcalc.split_x_y(ds)

total = S.shape[0]
batch = int(np.round(total / BATCHSIZE))

def mshape(X):
    # reshape input to be [samples, time steps, features]
    return np.reshape(X, (-1,  TIME_STEP, X.shape[1]))

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])        # shape(batch, 5, 1)
tf_y = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])        # input y

# RNN
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)
init_s = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)    # very first hidden state
outputs, final_s = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    tf_x,                       # input
    initial_state=init_s,       # the initial hidden state
    time_major=False,          # False: (batch, time step, input); True: (time step, batch, input)
)
outs2D = tf.reshape(outputs, [-1, CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
net_outs2D = tf.layers.dense(outs2D, INPUT_SIZE)
outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])          # reshape back to 3D

loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

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

sess = tf.Session()
sess.run(tf.global_variables_initializer())     # initialize var in graph
#summary_writer.add_summary(sess)

for epoch in range(10):
    for b in range(batch):
        start, end = b*(BATCHSIZE*TIME_STEP), (b+1)*(BATCHSIZE*TIME_STEP)   # time range
        if end >= total:
            end = total
        steps = np.linspace(start, end)
        x = S[start:end, ]  # shape (batch*time_step, input_size)
        y = T[start:end, ]  # shape (batch*time_step, input_size)

        x = mshape(x)  # shape (batch, time_step, input_size)
        y = mshape(y)  # shape (batch, time_step, input_size)

        if 'final_s_' not in globals():                 # first state, no any hidden state
            feed_dict = {tf_x: x, tf_y: y}
        else:                                           # has hidden state, so pass it to rnn
            feed_dict = {tf_x: x, tf_y: y, init_s: final_s_}
        _, pred_, final_s_ = sess.run([train_op, outs, final_s], feed_dict)     # train

        # plotting
        plt.plot(steps, y.flatten(), 'r-')
        plt.plot(steps, pred_.flatten(), 'b-')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.05)

plt.ioff()
plt.show()
