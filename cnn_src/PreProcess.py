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
    all_context, all_labels = zip(*loading_data_set(path))
    # all_context, all_labels = pickle.load(open("new_data.pkl", "rb"))
    train_context, test_context, train_labels, test_labels = train_test_split(all_context, all_labels, test_size=0.01)

    vocab_processor = learn.preprocessing.VocabularyProcessor(500, min_frequency=10)
    vocab_processor = vocab_processor.fit(all_context)

    print(datetime.datetime.now().isoformat())
    vec_train = list(vocab_processor.transform(train_context))
    vec_test = list(vocab_processor.transform(test_context))
    print(datetime.datetime.now().isoformat())
    print(len(vocab_processor.vocabulary_))
    pickle.dump([vec_train, vec_test, train_labels, test_labels], open('data.pkl', 'wb'))
    return vec_train, vec_test, train_labels, test_labels


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


def get_batch(batch_size):
    # x_train, x_test, y_train, y_test = get_one_hot('../文档')
    x_train, x_test, y_train, y_test = pickle.load(open("data.pkl", "rb"))
    data = list(zip(x_train, y_train))
    random.shuffle(data)

    for batch in range(0, len(data), batch_size):
        if batch + batch_size < len(data):
            yield data[batch: (batch + batch_size)]

if __name__ == "__main__":
    # get_one_hot('../文档')
    print("hello world")