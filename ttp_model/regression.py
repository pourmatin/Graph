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
import logging
from functools import lru_cache
# from price_fetcher.config import TICKERS
from ttp_model.dataset import Labels
from tensorflow.python.ops import rnn, rnn_cell
from ttp_model.utility import build

# logging.basicConfig(filename='info.log', level=logging.INFO)


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
            layer = tf.add(tf.matmul(activation, weights[i]), bias[i], name='logits')
            activation = tf.nn.sigmoid(layer)
        return layer

    def train(self):
        """
        the traing function
        :return: None
        """
        logits = self._model(self.xx)
        # loss = tf.sqrt(tf.reduce_mean(tf.square(self.yy - logits)))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.yy))
        optimize = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(1, self.hm_epochs + 1):
                for _ in range(self.n_datapoints // self._batch_size):
                    epoch_x, epoch_y = self.dataset.train.next_batch(self._batch_size)
                    sess.run(optimize, feed_dict={self.xx: epoch_x,
                                                  self.yy: epoch_y})
                loss_tr = sess.run(loss, feed_dict={self.xx: self.dataset.train.features,
                                                    self.yy: self.dataset.train.labels})
                print('Epoch {0}, Loss: {1}'.format(epoch, loss_tr))

            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self.yy, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', sess.run([logits, accuracy], {self.xx: self.dataset.test.features,
                                                             self.yy: self.dataset.test.labels}))
            saver.save(sess, "./my_model.ckpt")
            logging.info('Saved the model!')


class RnnModel:
    """
    The class to create and train the learning model
    """
    FINE = .001

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
        self.xx = tf.placeholder('float32', [None, 1, self.n_features], name='features')
        self.yy = tf.placeholder('float32', name='labels')

    @staticmethod
    def _weights(shape, name=None):
        initializer = tf.random_normal(shape)
        return tf.Variable(initializer, name=name)

    @lru_cache()
    def _model(self, data):
        layer = {'weights': tf.Variable(tf.random_normal([self._batch_size, self.n_labels])),
                 'biases': tf.Variable(tf.random_normal([self.n_labels]))}
        x = tf.transpose(data, [1, 0, 2])
        x = tf.reshape(x, [-1, self.n_features])
        x = tf.split(x, 1)

        lstm_cell = rnn_cell.LSTMCell(self._batch_size, name='basic_lstm_cell')
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'], name='rnn_model')

        return output

    def _cost(self, args):
        cost = 0
        logits = args[0]
        yy = args[1]
        if tf.argmax(yy, 1) == Labels.BUY:
            cost += self.FINE if tf.argmax(logits, 1) == Labels.HOLD else \
                2 * self.FINE if tf.argmax(logits, 1) == Labels.SELL else 0
        elif tf.argmax(yy, 1) == Labels.SELL:
            cost += self.FINE if tf.argmax(logits, 1) == Labels.BUY else 0
        else:
            cost += self.FINE if tf.argmax(logits, 1) == Labels.BUY else 0
        return cost

    def train(self):
        """
        the traing function
        :return: None
        """
        logits = self._model(self.xx)
        cost = tf.reduce_mean(tf.map_fn(self._cost, (logits, self.yy), dtype=tf.float32))
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.yy))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            for epoch in range(self.hm_epochs):
                epoch_loss = 0
                for _ in range(int(self.dataset.train.num_examples / self._batch_size)):
                    epoch_x, epoch_y = self.dataset.train.next_batch(self._batch_size)
                    epoch_x = epoch_x.reshape((self._batch_size, 1, self.n_features))

                    _, c = sess.run([optimizer, cost], feed_dict={self.xx: epoch_x, self.yy: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', self.hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self.yy, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Logits:', logits.eval({self.xx: self.dataset.test.features.reshape((-1, 1, self.n_features))}))
            print('Accuracy:',
                  accuracy.eval({self.xx: self.dataset.test.features.reshape((-1, 1, self.n_features)),
                                 self.yy: self.dataset.test.labels}))
            saver.save(sess, "./my_model.ckpt")
            logging.info('Saved the model!')


def main():
    """
    the main function
    :return: None
    """
    data = build(['BAC'],
                 pos_limit=.001,
                 path='./data.xlsx',
                 period=1200,
                 resample=1,
                 # start_date=datetime.date(2018,10,10)
                 )
    model = RnnModel(data, epochs=100, layers=4, batches=128)
    model.train()
    # model.test()


if __name__ == '__main__':
    main()
