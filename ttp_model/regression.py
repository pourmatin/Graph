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
import datetime
from tensorflow.python.ops import rnn, rnn_cell
from ttp_model.dataset import AcquireData
from price_fetcher.config import TICKERS


acquire = AcquireData(['AAPL'], pos_limit=.003, valid_ratio=0, train_ratio=.75)#, start_date=datetime.date(2018,9,6))
mnist = acquire.build()


hm_epochs = 50
n_classes = 3
batch_size = 128
chunck_size = 72
n_chuncks = 10
rnn_size = 200


xx = tf.placeholder('float', [None, n_chuncks, chunck_size], name='input')
yy = tf.placeholder('float')


def recurrnet_neural_network_model(data):
    """
    The main regression modeler
    :param data: the raw input data
    :return: the output labels
    """
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(data, [1, 0, 2])
    x = tf.reshape(x, [-1, chunck_size])
    x = tf.split(x, n_chuncks)

    lstm_cell = rnn_cell.LSTMCell(rnn_size, name='basic_lstm_cell')
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'], name='rnn_model')

    return output


def train_neural_network(data):
    """
    The ttp_model function
    :param data: input data
    :return: None
    """
    print('Starting...')
    prediction = recurrnet_neural_network_model(data)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=yy))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chuncks, chunck_size))
                _, c = sess.run([optimizer, cost], feed_dict={xx: epoch_x, yy: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yy, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({xx: mnist.test.features.reshape((-1, n_chuncks, chunck_size)),
                                          yy: mnist.test.labels}))
        # Save the variables to disk.
        save_path = saver.save(sess, "./my_model.ckpt")
        print("Model saved in path: %s" % save_path)


train_neural_network(xx)
