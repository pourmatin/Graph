# coding=utf-8
"""
PAT - the name of the current project.
event.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
6 / 15 / 18 - the current system date.
8: 03 AM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""

import datetime
import uuid
import logging


class Event:
    """
    objects of this class capture the trading events
    """
    def __init__(self, instrument, long=True):
        self._instrument = instrument
        self._long = long
        self.trades = []
        self.__close = False

    def __call__(self, size):
        self.open(size)
        return self

    @property
    def long(self):
        """
        Weather this trade is long or not
        :return bool:
        """
        return self._long

    @property
    def profits(self):
        """
        the current return of the trading events

        :return double: the return price
        """
        price = 0
        for trade in self.trades:
            if (trade.get('action') == 'sell' and self._long) or (trade.get('action') == 'buy' and not self._long):
                price += trade.get('price') * trade.get('size')
            else:
                price -= trade.get('price') * trade.get('size')
        return price

    def avg_price(self, action):
        """
        the average buy/sell price

        :param str action: buy or sell action
        :return double: price
        """
        price = 0
        counter = 0
        for trade in self.trades:
            if trade.get('action') == action:
                price += trade.get('price')
                counter += 1
        return price / counter

    @property
    def current_size(self):
        """
        the overal size of the trades

        :return int: number of shares
        """
        counter = 0
        for trade in self.trades:
            if trade.get('action') == 'buy':
                counter += trade.get('size')
            else:
                counter -= trade.get('size')
        return counter

    def return_rate(self):
        """
        return rate of the trading event

        :return double: return of the investment
        """
        if self.long:
            return (self.avg_price('sell') - self.avg_price('buy')) / self.avg_price('buy')
        else:
            return (self.avg_price('buy') - self.avg_price('sell')) / self.avg_price('sell')

    @property
    def instrument(self):
        """
        The instrument of the trade event
        :return: instrument object
        """
        return self._instrument

    def change(self):
        """
        current return rate of the trade
        :return: float
        """
        p1 = self.instrument.price
        p0 = self.trades[-1].

    def _execute(self, size, action):
        """
        method to open the trade event for this object. the next trades of this object will be appended

        :param size: the notional of the instrument (number of shares)
        :param str action: buy or sell
        :return str: The new trade ID
        """
        if self.__close:
            logging.error('Can not execute a trading event in a closed event object')
            return None
        current_size = self.current_size
        if (action == 'sell' and self._long) or (action == 'buy' and not self._long):
            new_size = current_size - size
        else:
            new_size = current_size + size
        if new_size < 0:
            raise Exception('Invalid action! size is greater than the current size in the Event object.')
        tradeid = str(uuid.uuid1())
        trade = {
                'price': self._instrument.price,
                'size': size,
                'time': datetime.datetime.now(),
                'action': action,
                'trade ID': tradeid}
        self.trades.append(trade)
        return tradeid

    def open(self, size):
        """
        Open the trading event by buying the shares or selling the shorted instruments

        :param size: the notional of the instrument (number of shares)
        :return str: The new trade ID
        """
        logging.info('Opening trading event {0}'.format(self))
        action = 'buy' if self._long else 'sell'
        return self._execute(size, action)

    def close(self):
        """
        Close the trading event by selling all of the shares or buying back the shorted instruments
        :return double: the total return of the series of the trading event
        """
        current_size = 0
        for trade in self.trades:
            current_size += trade.get('size')
        action = 'sell' if self._long else 'buy'
        self._execute(current_size, action)
        self.__close = True
        return self.profits()

    def add(self, size):
        """
        public method for adding a trade to the event

        :param size: the notional of the instrument (number of shares)
        :return str: The new trade ID
        """
        action = 'buy' if size > 0 else 'sell'
        if self.current_size + size < 0:
            logging.info('Closing trading event {0}'.format(self))
            return self.close()
        if not self.trades:
            raise Exception('The event object has not been opened yet. Use "open" instead.')
        return self._execute(size, action)
