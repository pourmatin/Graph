# coding=utf-8
"""
PAT - the name of the current project.
observer.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
6 / 18 / 18 - the current system date.
06: 17 PM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""

from instrument import Instrument
from publisher import create_topic
from config import PROJECT_ID
from google.cloud import pubsub_v1
import pickle
import pandas
import logging
import threading


# logging.basicConfig(filename='scanner.log', level=logging.INFO)

batch_settings = pubsub_v1.types.BatchSettings(
            max_bytes=1024,  # One kilobyte
            max_latency=1,  # One second
            max_messages=1000)


class PriceFetcher:
    """
    Obsersable class to scan and update the instruments
    """
    publisher = pubsub_v1.PublisherClient(batch_settings)

    def __init__(self, tickers, topic):
        self._tickers = tickers
        self.instruments = []
        self._topic = topic

    def _get_publisher_topic(self):
        """
        create a topic for each thread
        :return: None
        """
        ithread = threading.current_thread().name
        create_topic(PROJECT_ID, self._topic + ithread)

    def _get_score(self):
        tickers = [inst.ticker for inst in self.instruments]
        scores = [inst.score for inst in self.instruments]
        df = pandas.DataFrame({
                                'Ticker': tickers,
                                'Instrument': self.instruments,
                                'Score': scores
                            })
        df = df.sort_values('Score', ascending=False)
        return df

    @classmethod
    def _publish_messages(cls, topic_name, data):
        """
        Publishes multiple messages to a Pub/Sub topic.
        :param str topic_name: name of the topic
        :param data: the data to be published
        """
        topic_path = cls.publisher.topic_path(PROJECT_ID, topic_name)
        futures = []

        def _callback(msg_future):
            if msg_future.exception():
                logging.exception(
                    'Publishing message on {} threw an Exception {}.'.format(topic_name, msg_future.exception()))
            else:
                logging.info('Published message IDs:')
                logging.info(msg_future.result())

        try:
            dump = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            message_future = cls.publisher.publish(topic_path, data=dump)
            message_future.add_done_callback(_callback)
            futures.append(message_future)
        except Exception as e:
            logging.exception(e)

    def publish(self, inst):
        """
        publisher for the instruments of the object
        :param inst: list of instruments
        """
        ithread = threading.current_thread().name

        def _handle_exceptions(instrument):
            try:
                instrument.update()
                # logging.info('done with updates for thread {0}'.format(ithread))
                if instrument.changed:
                    instrument.write_live()
                    to_publish = instrument.latest_info
                    to_publish.update({'ticker': instrument.ticker})
                    self._publish_messages(self._topic + ithread, to_publish)
            except Exception as e:
                if 'connection' not in str(e).lower():
                    logging.error('Thread {0} is exiting!!'.format(ithread))
                    logging.exception(e)
                    exit(1)
            return instrument

        return [_handle_exceptions(inst) for inst in inst]

    def engine(self):
        """
        Starts scanning the instrumetns, creates events and adds closed events to the portfolio
        Only one of the parameters should be assigned.
        """
        instruments = [Instrument(ticker) for ticker in self._tickers]
        self._get_publisher_topic()
        while True:
            instruments = self.publish(instruments)
