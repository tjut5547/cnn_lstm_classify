import tensorflow as tf
import datetime
import numpy as np

from lstm import *
from PreProcess import *

import matplotlib.pyplot as plt
from tensorflow.contrib import learn

print("************************")

global_loss = []
global_accuracy = []


# 读取所有所有的数据
def get_new_data():
    neg = open(file="../data/negtive.neg", mode="r", encoding="utf-8")
    pos = open(file='../data/positive.pos', mode='r', encoding='utf-8')

    pos_context = []
    neg_context = []
    for a, b in zip(neg, pos):
        neg_context.append(a)
        pos_context.append(b)

    label_pos = [(0, 1) for _ in range(len(pos_context))]
    label_neg = [(1, 0) for _ in range(len(neg_context))]
    return np.array(neg_context + pos_context), np.array(label_neg + label_pos)


# 迭代器
def batch_iter(data, batch_size, num_epochs, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def split_data():
    x_text, y = get_new_data()
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]


    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.1 * float(len(y)))

    # 分割训练集个验证集
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        lstm = Model(num_layers=2,
                     seq_length=61,
                     vocab_size=30000,
                     rnn_size=128,
                     label_size=2)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.001).minimize(lstm.cost, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        def train_step(batch, label):
            feed_dict = {
                lstm.input_data: batch,
                lstm.targets: label,
                lstm.dropout_keep_prob: 0.5
            }
            _, step, loss, accuracy = sess.run(
                [optimizer, global_step, lstm.cost, lstm.accuracy],
                feed_dict=feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, accuracy {}".format(time_str, step, loss, accuracy))
            global_loss.append(loss)
            global_accuracy.append(accuracy)

        def dev_step(batch, label):
            feed_dict = {
                lstm.input_data: batch,
                lstm.targets: label,
                lstm.dropout_keep_prob: 0.5
            }
            step, loss, accuracy = sess.run([global_step, lstm.cost, lstm.accuracy], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, accuracy {}".format(time_str, step, loss, accuracy))


        x_train, y_train, x_dev, y_dev = split_data()
        batches = batch_iter(list(zip(x_train, y_train)), batch_size=200, num_epochs=50)
        for data in batches:
            x_train, y_train = zip(*data)
            train_step(x_train, y_train)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
                print("")

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

