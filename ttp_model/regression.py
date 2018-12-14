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
import numpy as np
import pandas as pd
import os
# from price_fetcher.config import TICKERS
from tensorflow.python.ops import rnn, rnn_cell
from ttp_model import utility

# logging.basicConfig(filename='info.log', level=logging.INFO)


class RnnModel:
    """
    The class to create and train the learning model
    """

    def __init__(self, dataset, tickers, bins, epochs=10, lstm_size=128, learning_rate=.01):
        self.dataset = dataset
        self.tickers = tickers
        self.hm_epochs = epochs
        self.n_features = dataset.train.features.shape[1]
        self.n_labels = dataset.train.labels.shape[1]
        self.n_datapoints = dataset.train.features.shape[0]
        self._lstm_size = lstm_size
        self._lr = learning_rate
        self._keep_prob = 0.8
        self.xx = None
        self.yy = None
        self.bins = bins

    def build_lstm_graph(self):
        """
        Build the lstm graph without the input data
        :return: the graph
        """
        tf.reset_default_graph()
        lstm_graph = tf.Graph()

        with lstm_graph.as_default():
            self.xx = tf.placeholder('float32', [None, 1, self.n_features], name='features')
            self.yy = tf.placeholder('float32', name='labels')
            self.bins = tf.constant(self.bins, name='bins')
            with tf.name_scope("output_layer"):
                weight = tf.Variable(tf.random_normal([self._lstm_size, self.n_labels]), name='weights')
                biases = tf.Variable(tf.random_normal([self.n_labels]), name='biases')
                x = tf.transpose(self.xx, [1, 0, 2])
                x = tf.reshape(x, [-1, self.n_features])
                x = tf.split(x, 1)

                lstm_cell = rnn_cell.LSTMCell(self._lstm_size, name='basic_lstm_cell')
                outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

                logits = tf.add(tf.matmul(outputs[-1], weight), biases, name='rnn_model')

                tf.summary.histogram("last_lstm_output", outputs[-1])
                tf.summary.histogram("weights", weight)
                tf.summary.histogram("biases", biases)

            with tf.name_scope("train"):
                correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self.yy, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name='accuracy')
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.yy),
                                      name='loss'
                                      )
                tf.train.AdamOptimizer().minimize(loss, name="loss_mse_adam_minimize")
                tf.summary.scalar("loss", loss)
                tf.summary.scalar("accuracy", accuracy)

            # Operators to use after restoring the model
            for op in [logits, loss]:
                tf.add_to_collection('ops_to_restore', op)

        return lstm_graph

    def train(self):
        """
        training function
        :return:
        """
        lstm_graph = self.build_lstm_graph()
        graph_name = "{0}_ticker{1}_lr{2}_lstm{3}_features{4}_epoch".format(self.tickers,
                                                                            self._lr,
                                                                            self._lstm_size,
                                                                            self.n_features,
                                                                            self.hm_epochs
                                                                            )

        print("Graph Name:", graph_name)

        with tf.Session(graph=lstm_graph) as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter('_logs/' + graph_name, sess.graph)
            writer.add_graph(sess.graph)

            graph = tf.get_default_graph()
            tf.global_variables_initializer().run()

            input = self.dataset.test.features.reshape((-1, 1, self.n_features))
            test_data_feed = {self.xx: input, self.yy: self.dataset.test.labels}

            loss = graph.get_tensor_by_name('train/loss:0')
            minimize = graph.get_operation_by_name('train/loss_mse_adam_minimize')
            prediction = graph.get_tensor_by_name('output_layer/rnn_model:0')
            accuracy = graph.get_tensor_by_name('train/accuracy:0')

            _summary = None
            for epoch in range(self.hm_epochs):
                n_batch = int(self.dataset.train.num_examples / self._lstm_size)
                for _ in range(n_batch):
                    epoch_x, epoch_y = self.dataset.train.next_batch(self._lstm_size)
                    epoch_x = epoch_x.reshape((self._lstm_size, 1, self.n_features))

                    train_loss, _ = sess.run([loss, minimize], feed_dict={self.xx: epoch_x, self.yy: epoch_y})

                if epoch % 10 == 0:
                    test_acc, _summary = sess.run([accuracy, merged_summary], test_data_feed)
                    print("Epoch {0}: {1}".format(epoch, test_acc))

                writer.add_summary(_summary, global_step=epoch)

            print("Final Results:")
            test_acc, final_loss = sess.run([accuracy, loss], test_data_feed)
            print(test_acc, final_loss)

            graph_saver_dir = os.path.join(MODEL_DIR, graph_name)
            if not os.path.exists(graph_saver_dir):
                os.mkdir(graph_saver_dir)

            saver = tf.train.Saver()
            saver.save(sess, os.path.join(graph_saver_dir, "stock_rnn_model_%s.ckpt" % graph_name),
                       global_step=self.hm_epochs)

        # with open("final_predictions.{}.json".format(graph_name), 'w') as fout:
        #     fout.write(json.dumps(final_prediction.tolist()))


            logit = tf.argmax(prediction, 1)
            nn = 120
            for i in range(nn):
                l, = sess.run([logit], feed_dict={self.xx: input})
                input = np.delete(input, -1, 2)
                input = np.insert(input, [0], l.reshape(-1, 1, 1), 2)

            df = pd.DataFrame()
            df['Real'] = np.argmax(self.dataset.test.labels, 1)[:121]
            df['Predict'] = input[0].reshape(-1)
            df.to_csv('./test.csv')
            # probability = tf.nn.softmax(prediction)
            # target = tf.argmax(self.yy, 1)
            # t, a, p = sess.run([target, logit, probability],
            #                 feed_dict={self.xx: self.dataset.test.features.reshape((-1, 1, self.n_features)),
            #                            self.yy: self.dataset.test.labels.reshape((-1, self.n_labels))})
            # pp = np.max(p, 1)
            # for i, j, k in zip(a, t, pp):
            #     print(i, j, k)
            #     if i != j:
            #         print('HERE!')
            target = self.dataset.test.labels
            target = np.delete(target, [i for i in range(nn)], 0)
            input = np.delete(input, [i for i in range(input.shape[0] - 1, input.shape[0]- nn - 1, -1)], 0)
            print('Accuracy:',
                  accuracy.eval({self.xx: input,
                                 self.yy: target}))


MODEL_DIR = './'
import datetime

def main():
    """
    the main function
    :return: None
    """
    utility.build()

    # dp = utility.DataProcessor(tickers,
    #                    # start_date=datetime.date(2018, 11, 20),
    #                    path='./data.xlsx'
    #                    )
    # data, bins = dp.build()
    # model = RnnModel(data, tickers=tickers, bins=bins, epochs=200, learning_rate=.001)
    # model.train()


if __name__ == '__main__':
    main()
