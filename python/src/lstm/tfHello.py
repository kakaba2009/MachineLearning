import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess  = tf.Session()

print(sess.run(hello))