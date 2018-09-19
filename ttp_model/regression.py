# coding=utf-8
"""
PAT - the name of the current project.
instrument.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
6 / 15 / 18 - the current system date.
8: 03 AM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""

import tensorflow as tf
# import numpy as np
from functools import lru_cache
# from tensorflow.python.ops import rnn, rnn_cell
from ttp_model.dataset import AcquireData
from price_fetcher.config import TICKERS


class Model:
    """
    The class to create and train the learning model
    """
    def __init__(self, dataset, epochs=10, nodes=None, layers=1, batches=100, learning_rate=.01):
        self.dataset = dataset
        self.hm_epochs = epochs
        self.n_features = dataset.train.features.shape[1]
        self.n_labels = dataset.train.labels.shape[1]
        self.n_nodes = nodes or [32] * layers
        self.n_layers = layers
        self.n_datapoints = dataset.train.features.shape[0]
        self._batch_size = batches
        self._lr = learning_rate
        self.xx = tf.placeholder('float32', [None, self.n_features], name='features')
        self.yy = tf.placeholder('float32', [None, self.n_labels], name='labels')

    @staticmethod
    def _weights(shape, name=None):
        initializer = tf.random_normal(shape)
        return tf.Variable(initializer, name=name)

    @lru_cache()
    def _model(self, data):
        dim = [self.n_features] + self.n_nodes + [self.n_labels]
        weights = [self._weights([dim[i], dim[i + 1]]) for i in range(len(dim) - 1)]
        bias = [self._weights([dim[i + 1]]) for i in range(len(dim) - 1)]

        activation = data
        layer = None
        for i in range(len(dim) - 1):
            layer = tf.add(tf.matmul(activation, weights[i]), bias[i])
            activation = tf.nn.sigmoid(layer)
        return layer

    def train(self):
        """
        the traing function
        :return: None
        """
        logits = self._model(self.xx)
        loss = tf.sqrt(tf.reduce_mean(tf.square(self.yy - logits)))
        # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.yy))
        optimize = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(1, self.hm_epochs + 1):
                for i in range(0, self.n_datapoints):
                    sess.run(optimize, feed_dict={self.xx: self.dataset.train.features[i: i + self._batch_size],
                                                  self.yy: self.dataset.train.labels[i: i + self._batch_size]})
                loss_tr = sess.run(loss, feed_dict={self.xx: self.dataset.train.features,
                                                    self.yy: self.dataset.train.labels})
                print('Epoch {0}, Loss: {1}'.format(epoch, loss_tr))

            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self.yy, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', sess.run([logits, accuracy], {self.xx: self.dataset.test.features,
                                                   self.yy: self.dataset.test.labels}))


def main():
    """
    the main function
    :return: None
    """
    acquire = AcquireData(['AAPL', 'BAC'],
                          pos_limit=.001,
                          valid_ratio=0,
                          train_ratio=.75,
                          # path='./data.xlsx'
                          )  # , start_date=datetime.date(2018,9,12))
    data = acquire.build()
    model = Model(data, layers=4, batches=128)
    model.train()
    # model.test()


if __name__ == '__main__':
    main()

# def recurrnet_neural_network_model(data):
#     """
#     The main regression modeler
#     :param data: the raw input data
#     :return: the output labels
#     """
#     layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
#              'biases': tf.Variable(tf.random_normal([n_classes]))}
#     x = tf.transpose(data, [1, 0, 2])
#     x = tf.reshape(x, [-1, chunck_size])
#     x = tf.split(x, n_chuncks)
#
#     lstm_cell = rnn_cell.LSTMCell(rnn_size, name='basic_lstm_cell')
#     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#     output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'], name='rnn_model')
#
#     return output


# def train_neural_network(data):
#     """
#     The ttp_model function
#     :param data: input data
#     :return: None
#     """
#     print('Starting...')
#     prediction = recurrnet_neural_network_model(data)
#     # OLD VERSION:
#     # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
#     # NEW:
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=yy))
#     optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(cost)
#
#     # Add ops to save and restore all the variables.
#     saver = tf.train.Saver()
#
#     with tf.Session() as sess:
#         # OLD:
#         # sess.run(tf.initialize_all_variables())
#         # NEW:
#         sess.run(tf.global_variables_initializer())
#
#         for epoch in range(hm_epochs):
#             epoch_loss = 0
#             for _ in range(int(mnist.train.num_examples / batch_size)):
#                 epoch_x, epoch_y = mnist.train.next_batch(batch_size)
#                 epoch_x = epoch_x.reshape((batch_size, n_chuncks, chunck_size))
#                 _, c = sess.run([optimizer, cost], feed_dict={xx: epoch_x, yy: epoch_y})
#                 epoch_loss += c
#
#             print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
#
#         correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yy, 1))
#         pred = tf.maximum(prediction, 1)
#
#         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#         print('Accuracy:', sess.run([prediction, yy, accuracy], {xx: mnist.test.features.reshape((-1,
# n_chuncks, chunck_size)),
#                                           yy: mnist.test.labels}))
#         # Save the variables to disk.
#         save_path = saver.save(sess, "./my_model.ckpt")
#         print("Model saved in path: %s" % save_path)
#
#
# train_neural_network(xx)
