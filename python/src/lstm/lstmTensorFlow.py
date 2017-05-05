from __future__ import print_function

import os
import warnings
import numpy as np
import src.mylib.mcalc as mcalc
import src.mylib.mlstm as mlstm
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

loaded = False
modelSaved = "../model/JPYRMSPropLinear12x6xD6.h5"
graphSaved = "../model/JPYRMSPropLinear12x6xD6.html"

batch_size = 128
epochs_num = 1
output_dim = 6

np.random.seed(6)  # fix random seed

ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 1000)
ds = ds[['Close']]

P = mcalc.m_pct(ds, True)

T = mcalc.vector_delay_embed(P, output_dim, 1)

X, Y = mcalc.split_x_y(T)

def mshape(X):
    # reshape input to be [samples, time steps, features]
    return np.reshape(X, (X.shape[0],  -1, X.shape[1]))

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

kf = KFold(n_splits=3, shuffle=False, random_state=None)

# Parameters
learning_rate = 0.001
display_step = 10

# Network Parameters
n_steps  = 1   # timesteps
n_hidden = 16  # hidden layer num of features

# tf Graph input
x = tf.placeholder("float", [None, n_steps, output_dim])
#tf.summary.scalar("x", x)
y = tf.placeholder("float", [None, output_dim])
#tf.summary.scalar("y", y)

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, output_dim]))
}
biases = {
    'out': tf.Variable(tf.random_normal([output_dim]))
}

def RNN(name, x, weights, biases):
    with tf.name_scope(name):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        model = tf.matmul(outputs[-1], weights['out']) + biases['out']
        tf.summary.histogram(name, model)
        return model

pred = RNN("lstm1", x, weights, biases)
tf.summary.histogram("pred", pred)

# Define loss and optimizer
cost = tf.losses.mean_squared_error(labels=y, predictions=pred)
tf.summary.scalar("cost", cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.contrib.metrics.streaming_mean_squared_error (pred, y)

# Initializing the variables
init = tf.global_variables_initializer()

merged = tf.summary.merge_all()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step <= epochs_num:
        batch_x, batch_y = X, Y
        # Reshape data to get 1 seq of 6 elements
        batch_x = mshape(batch_x)
        # Run optimization op (backprop)
        summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(acc, step)
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        train_writer = tf.summary.FileWriter('../log', sess.graph)
        train_writer.add_summary(summary, step)
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
        # sess.run(accuracy, feed_dict={x: test_data, y: test_label}))