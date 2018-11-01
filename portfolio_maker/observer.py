# coding=utf-8
"""
PAT - the name of the current project.
subscriber.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
8 / 6 / 18 - the current system date.
10: 03 PM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""

from google.cloud import pubsub_v1
from price_fetcher.bigquery import GoogleQuery
from ttp_model.dataset import Labels
from ttp_model.utility import process_query, add_feature
import pickle
import pandas as pd
import tensorflow as tf
import datetime
import copy
import logging


STOP_LOSS = -.001
TARGET = .001


class Rnn:
    """
    predictor class
    """
    def __init__(self):
        self.chunck_size = 2
        self.n_chuncks = 1
        self.sess = tf.Session()
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('ttp_model/my_model.ckpt.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('ttp_model'))
        graph = tf.get_default_graph()
        self.xx = graph.get_tensor_by_name("features:0")
        self.logits = graph.get_tensor_by_name("rnn_model:0")
        self.prediction = tf.argmax(self.logits, 1)
        # print(self.prediction.shape)

    def predict(self, data):
        """

        :param data:
        :return:
        """
        feed_dict = {self.xx: data.reshape((1, 1, -1))}
        decition = self.sess.run(self.prediction, feed_dict)
        return decition


class Observer:
    """
    class that observes price changes and informs broker when to buy and sell
    """
    def __init__(self, tickers, start_date=None):
        self.tickers = tickers
        self._start = start_date
        self.instruments = {}
        self._freq = 10
        self.rnn = Rnn()
        self._events = {}

    def initiate(self):
        """

        :return:
        """
        for ticker in self.tickers:
            google = GoogleQuery(ticker, dataset_id='my_dataset')
            query = google.query(start=self._start, last=3600)
            query = query.sort_index()
            if query.empty:
                continue
            dates = list(set(query.index.date))
            print(ticker)
            result = pd.DataFrame(columns=['Price', 'Volume'])
            for date in dates:
                q_start = query[query.index.date == date].index[0]
                start = max([datetime.datetime(date.year, date.month, date.day, 8, 30), q_start])
                end = datetime.datetime(date.year, date.month, date.day, 15, 0)
                if date == datetime.date.today():
                    end = min([end, datetime.datetime.now() - datetime.timedelta(seconds=300)])
                df = pd.DataFrame(index=pd.date_range(start=start, end=end, freq=str(self._freq) + 'S'),
                                  columns=['Price', 'Volume'])
                df.index.name = 'Time'
                df = pd.merge_asof(df, query.sort_index(), left_index=True, right_index=True)
                df = df.rename(columns={'Price_y': 'Price', 'Volume_y': 'Volume'})
                df = df.drop([col for col in df.columns if col.endswith('_x')], axis=1)
                result = result.append(df)
            result = result.sort_index().tail(3600//self._freq)
            self.instruments.update({ticker: result})

    def receive_messages(self, project, subscription_name):
        """Receives messages from a pull subscription."""
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(project, subscription_name)

        def _callback(message):
            data = pickle.loads(message.data)
            ticker = data.get('Ticker')
            current_position = self._update(data)
            # print('Received message: {}'.format(current_position))
            event = self._events.get(ticker)
            if current_position is not None:
                label = self.rnn.predict(current_position)
                if label == Labels.BUY and not event:
                    event.open(1)
                    logging.info('Buying {0} at {1} at price {2}'.format(ticker, data.get('Time'), data.get('Price')))
                elif label != Labels.BUY and event and event.change < STOP_LOSS or event.change > TARGET:
                    event.close()
                    logging.info('Selling {0} at {1} at price {2}'.format(ticker, data.get('Time'), data.get('Price')))
            if message.attributes:
                print('Attributes:')
                for key in message.attributes:
                    value = message.attributes.get(key)
                    print('{}: {}'.format(key, value))
            message.ack()

        # Limit the subscriber to only have ten outstanding messages at a time.
        flow_control = pubsub_v1.types.FlowControl(max_messages=10)
        subscriber.subscribe(subscription_path, callback=_callback, flow_control=flow_control)

        # Blocks the thread while messages are coming in through the stream. Any
        # exceptions that crop up on the thread will be set on the future.
        # while True:
        #     time.sleep(60)

    def _update(self, data):
        ticker = data.pop('Ticker')
        df = self.instruments.get(ticker)
        if df is not None:
            new_row = pd.DataFrame.from_dict({'Time': [datetime.datetime.now()],
                                              'Price': data.get('Price'),
                                              'Volume': data.get('Volume')}
                                             ).set_index('Time')
            df = df.append(new_row).sort_index()
            df = df.tail(3600)
            self.instruments.update({ticker: df})
            frame = copy.deepcopy(df)
            frame['Ticker'] = ticker
            frame = process_query(frame)
            frame = add_feature(frame)
            features = [col for col in frame.columns if col.startswith('F') or col.startswith('dF')]
            frame = frame[features]
            last_row = frame.tail(1).values.astype(float)
            # print(last_row)
            return last_row
