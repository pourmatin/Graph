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
import pandas
import datetime
import pytz
import logging
from requests import exceptions
from bigquery import GoogleQuery
from stock_api import Iex


logger = logging
logging.basicConfig(filename='ERRORS.log', level=logging.ERROR)


class Instrument:
    """
    base class for Stocks
    """
    def __init__(self, name):
        self.ticker = name
        self._stats = None
        self._financials = None
        self._news = None
        self._price = None
        self._score = None
        self._iex = Iex('iexfinance', self.ticker)
        self._price_history = None
        self.gquery = GoogleQuery(ticker=name, dataset_id='my_dataset', table_id='live_' + name)
        self.latest_info = {'Price': None, 'Volume': None}
        self.changed = True
        self._timezone = 'America/Chicago'

    @property
    def timezone(self):
        """
        the time zone of the user
        :return: str time zone
        """
        return self._timezone

    @timezone.setter
    def timezone(self, value):
        self._timezone = value

    @staticmethod
    def _append(df, time, price, volume):
        new_row = [{'Time': time, 'Price': price, 'Volume': volume}]
        new_df = pandas.DataFrame(new_row).set_index('Time')
        df = df.append(new_df)
        return df

    def _snapshot(self, price, volume):
        local_dt = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)\
            .astimezone(tz=pytz.timezone(self.timezone))
        local_dt = local_dt.replace(tzinfo=None)
        if self._price_history is not None:
            df = self._append(self._price_history, local_dt, price, volume)
        else:
            df = self._init_price_hist(local_dt, price, volume)
        self._price_history = df

    def _init_price_hist(self, time, price, volume):
        df = pandas.DataFrame(index=pandas.date_range(end=time, periods=3600, freq='1S'),
                              columns=['Price', 'Volume'])
        df.index.name = 'Time'
        query = self.gquery.query(last=3600)
        if not query.empty:
            df = pandas.merge_asof(df, query.sort_index(), left_index=True, right_index=True)
            df = df.rename(columns={'Price_y': 'Price', 'Volume_y': 'Volume'})
            df = df.drop([col for col in df.columns if col.endswith('_x')], axis=1)
        df = self._append(df, time, price, volume)
        return df.tail(3600)

    @property
    def price(self):
        """
        the current, or the latest price of the stock
        :return:
        """
        return self._price

    @property
    def stats(self):
        """
        statistics of the instrument in the format of dictionary
        :return: dictionary
        """
        return self._stats

    @property
    def financials(self):
        """
        financial properties of the stock
        :return: dictionary
        """
        return self._financials

    @property
    def news(self):
        """
        recent news related to the stock
        :return: dictionary
        """
        return self._news

    def update(self):
        """
        gets the price, stats, news and financial updates from the server and calls the setters

        """
        try:
            price = self._iex.price()
            vol = self._iex.volume()
            self.changed = self.latest_info.get('Price') != price
            if self.changed:
                self._snapshot(price, vol)
                self.latest_info.update({'Price': price, 'Volume': vol})
        except (ConnectionError, SystemError, exceptions.ConnectionError):
            self.changed = False
        # stats = obj.stats()
        # finace = obj.financials()
        # news = obj.news()

    @property
    def score(self):
        """
        the score of the intrument is an indicator of how likely it is for the price to increase or decrease

        :return double: a number from 0 to 100
        """
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    def snapshot(self, rule):
        """
        takes a snapshot of the price qoute
        :param str rule: the sampling size (e.g. 1m)
        """
        sample = self._price_history.resample(rule).agg({'Price': 'ohlc', 'Volume': 'mean'})
        sample.columns = sample.columns.droplevel()
        return sample[sample.open > 0]

    def write_ohlc(self, upto=None):
        """
        writes the historical prices of the instrument object into the database
        :param datetime upto: write upto the specified time
        :return: list of errors if any
        """
        upto = upto or datetime.datetime.now()
        rows = []
        tbl = self.snapshot('60S')
        for row in tbl.iterrows():
            if row[0] < upto:
                rows.append((row[0], self.ticker, row[1][0], row[1][1], row[1][2], row[1][3], int(row[1][4])))
            else:
                break
        try:
            errors = self.gquery.write(rows, self.ticker)
        except Exception as errors:
            logging.exception(errors)
        else:
            self._price_history = self._price_history[self._price_history['Time'] > upto]
            if errors:
                logging.exception(errors)

    def write_live(self):
        """
        writes the historical prices of the instrument object into the database
        :return: list of errors if any
        """
        last_row = self._price_history.get_values()[-1].tolist()
        last_time = [self._price_history.index[-1]]
        last_time.extend(last_row)
        # last_row.insert(1, self.ticker)
        last_row = [tuple(last_time)]
        try:
            self.gquery.write(last_row)
        except (ConnectionError, SystemError):
            # logging.exception(e)
            pass
