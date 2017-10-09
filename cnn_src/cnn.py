import tensorflow as tf


class Cnn(object):
    def __init__(self, sequence_length, vocab_size, embedding_size,
                 filter_sizes, num_filters, num_classes, number_sample):
        self.label = tf.placeholder(tf.float16, [number_sample, num_classes], name="label")
        self.input_sentence = tf.placeholder(tf.int32, [number_sample, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W",
                dtype=tf.float32)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_sentence)
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-max-pool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                W2 = tf.Variable(tf.truncated_normal([filter_size, 1, 1, num_filters], stddev=0.1), name="W")
                W3 = tf.Variable(tf.truncated_normal([filter_size, 1, 1, num_filters], stddev=0.1), name="W")
                W4 = tf.Variable(tf.truncated_normal([filter_size, 1, 1, num_filters], stddev=0.1), name="W")
                conv1 = tf.nn.conv2d(input=self.embedded_chars_expand,
                                     filter=W1,
                                     strides=[1, 1, 1, 1],
                                     padding="VALID")

                conv2 = tf.nn.conv2d(input=conv1,
                                     filter=W2,
                                     strides=[1, 1, 1, 1],
                                     padding="VALID")

                conv3 = tf.nn.conv2d(input=conv2,
                                     filter=W3,
                                     strides=[1, 1, 1, 1],
                                     padding="VALID")

                conv4 = tf.nn.conv2d(input=conv3,
                                     filter=W4,
                                     strides=[1, 1, 1, 1],
                                     padding="VALID")

                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                relu = tf.nn.relu(tf.nn.bias_add(conv4, b), name="relu")
                pooled = tf.nn.max_pool(
                    relu,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)

        # 句子的特征向量表示
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("full_connected_layer"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.score = tf.nn.xw_plus_b(self.h_drop, W, b)

        # with tf.name_scope("softmax"):
        #     self.result = tf.nn.softmax(logits=self.full_connected, dim=1, name='softmax')

        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.score, axis=1, name="predictions")
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.score)
            self.losses = tf.reduce_mean(losses) + 0.03 * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        tensorlist = [self.accuracy, self.score, self.losses]
        for tensor in tensorlist:
            print(tensor)


# cnn = Cnn(sequence_length=35,
#           vocab_size=1000,
#           embedding_size=50,
#           filter_sizes=[1, 2, 3, 4, 5],
#           num_filters=5,
#           num_classes=9,
#           number_sample=200)
