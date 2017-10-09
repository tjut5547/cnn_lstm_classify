# -*- coding:utf-8 -*-
# !/usr/bin/env python

import pickle
import os
import time
import datetime
import jieba
import jieba.analyse
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn

label_list = ['C3-Art', 'C31-Enviornment', 'C32-Agriculture', 'C34-Economy',
              'C38-Politics', 'C39-Sports', 'C7-History']

stopword = set()
fd = open('../stopwords.txt', 'r', encoding='gbk')
for line in fd:
    stopword.add(line.strip())


def remove_stop_word(article):
    new_article = []
    for word in article:
        if word not in stopword:
            new_article.append(word)
    return new_article


def soft_max_label(label):
    new_label = 7 * [0]
    index = label_list.index(label)
    new_label[index] = 1
    return new_label


def get_one_hot(path):
    all_context, all_labels = zip(*pickle.load(open("new_data.pkl", "rb")))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=2000, min_frequency=20)
    all_context = np.array(list(vocab_processor.fit_transform(all_context)))
    all_labels = np.array(list(all_labels))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_labels)))
    x_shuffled = all_context[shuffle_indices]
    y_shuffled = all_labels[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.1 * float(len(all_labels)))

    # 分割训练集个验证集
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev


def delete_and_split(all_context, all_labels):
    new_data = []
    data = zip(all_context, all_labels)
    for context, label in data:
        article = remove_stop_word(jieba.cut(context))
        string = ' '.join(article)
        new_data.append((string, soft_max_label(label)))
    return new_data
            

def loading_data_set(path):
    all_context = []
    all_labels = []
    all_directory = os.listdir(path)
    for directory in all_directory:
        if directory not in label_list:
            continue
        all_file = os.listdir(os.path.join(path, directory))
        print("路径 = ", directory, "时间：", time.asctime((time.localtime(time.time()))))
        for file in all_file:
            with open(os.path.join(path, directory, file), 'r', encoding='gbk') as fd:
                context = fd.read()
            all_context.append(context)
            all_labels.append(directory)

    print("分词开始时间：", time.asctime((time.localtime(time.time()))))
    new_data = delete_and_split(all_context, all_labels)
    print("分词结束时间：", time.asctime((time.localtime(time.time()))))
    pickle.dump(new_data, open("new_data.pkl", "wb"))
    return new_data


def get_batch(epoches, batch_size):
    _, _, all_context, all_labels = get_one_hot('../文档')
    data = list(zip(all_context, all_labels))
    for epoch in range(epoches):
        random.shuffle(data)
        for batch in range(0, len(data), batch_size):
            if batch + batch_size < len(data):
                yield data[batch: (batch + batch_size)]

if __name__ == "__main__":
    get_one_hot('../文档')
