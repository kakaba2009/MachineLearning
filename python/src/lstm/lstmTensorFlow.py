from __future__ import print_function

import os
import time
import math
import warnings
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import src.mylib.mlstm as mlstm
import src.mylib.mcalc as mcalc
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

tf.set_random_seed(0)  # fix random seed

# Hyper Parameters
EPOCHSIZE = 25      # rnn epoch size
BATCHSIZE = 10      # rnn batch size
TIME_STEP = 5       # rnn time step
CELLSIZE = 32       # rnn cell size
FEATURES = 1        # rnn input size
DROPOUTS = 0.5

max_learning_rate = 0.001
min_learning_rate = 0.00001
decay_speed = 1000.0

ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 1000)
ds = ds[['Close']].values

S, T = mcalc.split_x_y(ds)

total = S.shape[0]

def mshape(X):
    # reshape input to be [samples, time steps, features]
    return np.reshape(X, (-1,  TIME_STEP, FEATURES))

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP, FEATURES])        # shape(batch, steps, feature)
tf_y = tf.placeholder(tf.float32, [None, TIME_STEP, FEATURES])        # input y
# variable learning rate
lr = tf.placeholder(tf.float32)

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=CELLSIZE, forget_bias=1.0, state_is_tuple=True)
rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=DROPOUTS)
init_s = rnn_cell.zero_state(batch_size=BATCHSIZE, dtype=tf.float32)    # very first hidden state
outputs, final_s = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    tf_x,                       # input
    initial_state=init_s,       # the initial hidden state
    time_major=False,          # False: (batch, time step, input); True: (time step, batch, input)
)
outs2D = tf.reshape(outputs, [-1, CELLSIZE])                     # reshape 3D output to 2D for fully connected layer
net_outs2D = tf.layers.dense(outs2D, FEATURES)
outs = tf.reshape(net_outs2D, [-1, TIME_STEP, FEATURES])         # reshape back to 3D

loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)  # compute cost
tf.summary.scalar("loss", loss)

train_op = tf.train.AdamOptimizer(lr).minimize(loss)

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     # initialize var in graph
    saver.restore(sess, "checkpoints/lstmTensorFlow.ckpt")
    print("lstmTensorFlow.ckpt restored")

    summary_writer = tf.summary.FileWriter("../log/", sess.graph)

    #plt.figure(1, figsize=(12, 5))
    #plt.ion()

    steps = 0
    for epoch in range(EPOCHSIZE):
        for x, y in tl.iterate.seq_minibatches(inputs=S, targets=T, batch_size=BATCHSIZE, seq_length=TIME_STEP, stride=1):
            steps += 1

            x = mshape(x)  # shape (batch, time_step, input_size)
            y = mshape(y)  # shape (batch, time_step, input_size)
            print(steps)
            #print("x=", x.flatten())  # shape (batch*time_step, input_size)
            #print("y=", y.flatten())  # shape (batch*time_step, input_size)

            # learning rate decay
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-steps / decay_speed)

            if 'final_s_' not in globals():                # first state, no any hidden state
                feed_dict = {tf_x: x, tf_y: y, lr: learning_rate}
            else:                                           # has hidden state, so pass it to rnn
                feed_dict = {tf_x: x, tf_y: y, lr: learning_rate, init_s: final_s_}

            _, pred_, final_s_ = sess.run([train_op, outs, final_s], feed_dict)     # train

            if steps % 2 == 0:
                # Calculate batch loss
                bloss, summary = sess.run([loss, merged], feed_dict={tf_x: x, tf_y: y, lr: learning_rate})
                print("Iter " + str(steps) + ", Minibatch Loss= " + "{:.6f}".format(bloss))
                summary_writer.add_summary(summary, steps)

            # plotting
            #plt.plot(y.flatten(), 'r-')
            #plt.plot(pred_.flatten(), 'b-')
            #plt.draw()
            #plt.pause(0.05)
        save_path = saver.save(sess, "checkpoints/lstmTensorFlow.ckpt")
        print("checkpoints saved:", save_path)

    #plt.ioff()
    #plt.show()
