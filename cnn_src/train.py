import tensorflow as tf
import datetime
import numpy as np

from cnn import *
from PreProcess import *

import matplotlib.pyplot as plt

global_loss = []
global_accuracy = []
batches = get_batch(50)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = Cnn(sequence_length=500,
                  vocab_size=56322,
                  embedding_size=100,
                  filter_sizes=[2, 3, 4, 5, 6, 7],
                  num_filters=150,
                  num_classes=7,
                  number_sample=50)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.0005).minimize(cnn.losses, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        def train_step(batch, label):
            feed_dict = {
                cnn.input_sentence: batch,
                cnn.label: label,
                cnn.dropout_keep_prob: 0.001
            }
            _, step, loss, result, accuracy, predictions = sess.run(
                [optimizer, global_step, cnn.losses, cnn.score, cnn.accuracy, cnn.predictions],
                feed_dict=feed_dict)

            print(predictions)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, accuracy {}".format(time_str, step, loss, accuracy))
            global_loss.append(loss)
            global_accuracy.append(accuracy)

        def dev_step(batch, label):
            feed_dict = {
                cnn.input_sentence: batch,
                cnn.label: label,
                cnn.dropout_keep_prob: 0.5
            }
            step, loss = sess.run([global_step, cnn.losses], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

        for data in batches:
            x_train, y_train = zip(*data)
            train_step(x_train, y_train)
            current_step = tf.train.global_step(sess, global_step)

        x = list(range(len(global_loss)))
        plt.plot(x, global_loss, 'r', label="loss")
        plt.xlabel("batches")
        plt.ylabel("loss")
        plt.savefig("loss_modify.png")
        plt.close()

        plt.plot(x, global_accuracy, 'b', label="accuracy")
        plt.xlabel("batches")
        plt.ylabel("accuracy")
        plt.savefig("accuracy.png")
        plt.close()

