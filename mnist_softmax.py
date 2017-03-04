# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import pickle

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=0)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print("Training data samples: {}".format(len(mnist.train.images)))
    print("Test data samples: {}".format(len(mnist.test.images)))

    # Create the model
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784])

    with tf.name_scope("convolution_1"):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope("max_pool_1"):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("convolution_2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope("max_pool_2"):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope("feature_extraction"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

    with tf.name_scope("output"):
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        prediction = tf.argmax(y_conv, 1)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        tf.summary.scalar("cross_entropy", cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Train
    tf.set_random_seed(1)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./log/train", graph=sess.graph)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        summ, _ = sess.run([merged, train_step],
                           feed_dict={x: batch_xs, y_: batch_ys,
                                      keep_prob: 0.5})
        writer.add_summary(summ, i)
        if (i + 1) % 100 == 0:
            summ, accu = sess.run(
                [merged, accuracy],
                feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("Step {:4d}, training accuracy {:.4f}".format(i + 1, accu))

    # Test trained model
    prediction = sess.run(tf.argmax(y_conv, 1),
                          feed_dict={
                              x: mnist.test.images, keep_prob: 1.0})
    with open("test_result.pkl", "wb") as fp:
        pickle.dump((prediction,
                     mnist.test.images,
                     mnist.test.labels.argmax(axis=1)), fp, -1)
    num_correct = (prediction == mnist.test.labels.argmax(axis=1)).sum()
    print("Accuracy: {:.4f}".format(num_correct / len(prediction)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
