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
# import time
import pickle


class Observer:
    """
    class that observes price changes and informs broker when to buy and sell
    """
    def __init__(self, tickers, strategy=None):
        self.tickers = tickers
        self.strategy = strategy
        self.instruments = {}

    def receive_messages(self, project, subscription_name):
        """Receives messages from a pull subscription."""
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(project, subscription_name)

        def _callback(message):
            print('Received message: {}'.format(message.data))
            instrument = pickle.loads(message.data)
            instrument.score = self.strategy.score(instrument)
            self.instruments.update({instrument.ticker: instrument})
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

    def _get_trades(self, scores, threshhold):
        events = {}
        long = scores[scores['Score'] > 70]
        long_sum = long.apply(np.sum) - 70. * len(long)

        for row in long.iterrows():
            size = int((row.score - 70.) / long_sum * self._balance / row.Instrument.price)
            if size < threshhold:
                continue
            if not events.get(row.Ticker, None):
                events.update({row.Ticker: Event(row.Instrument)})
                events.get(row.Ticker).open(size)
            elif not -threshhold < (events.get(row.Ticker).size() - size) < threshhold:
                new_size = size - events.get(row.Ticker).size()
                events.get(row.Ticker).add(new_size)
        return events
