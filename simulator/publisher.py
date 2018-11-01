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

from price_fetcher.bigquery import GoogleQuery
from price_fetcher.observer import PriceFetcher
from price_fetcher.publisher import create_topic
from price_fetcher.config import PROJECT_ID, TICKERS
import pandas as pd
import datetime
import logging


class Simulator:
    """
    simulator class for publishing prices in real time
    """
    def __init__(self, tickers, start_date, end_date=None, topic=None):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date or start_date
        self._topic = topic or 'simulator'

    def get_data(self):
        """
        get the price history from google query
        :return: dataframe of tickers and price history
        """
        instruments = pd.DataFrame()
        for ticker in self.tickers:
            google = GoogleQuery(ticker, dataset_id='my_dataset')
            history = google.query(start=self.start)
            print(ticker)
            if not history.empty:
                history = history[history.index.date <= self.end]
            history['Ticker'] = ticker
            logging.info('imported {0}'.format(ticker))
            instruments = instruments.append(history)
        return instruments.sort_index()

    def engine(self):
        """
        runs for the same period of time and publishes the queried data
        :return: None
        """
        fetcher = PriceFetcher(tickers=self.tickers, topic=self._topic)
        create_topic(PROJECT_ID, self._topic)
        instruments = self.get_data()
        dt = datetime.datetime.now()
        delta = dt -instruments.index[0]
        while True:
            dt = datetime.datetime.now()
            subset = instruments[instruments.index < dt - delta]
            for ticker in self.tickers:
                df = subset[subset.Ticker == ticker]
                if not df.empty:
                    data = {'Ticker': ticker,
                            'Time': df.iloc[-1].name,
                            'Price': df.iloc[-1].Price,
                            'Volume': df.iloc[-1].Volume}
                    fetcher._publish_messages(self._topic, data)
                    print(data)
            instruments = instruments[instruments.index >= dt - delta]
        # for row in instruments.itertuples():
        #     values = [item for item in row]
        #     data = dict((field, value) for field,value in zip(row._fields, values))
        #     fetcher._publish_messages(self._topic, data)


def main():
    """
    main function
    :return:
    """
    sim = Simulator(['BAC'], start_date=datetime.date(2018,10,22))
    sim.engine()


if __name__ == '__main__':
    main()
