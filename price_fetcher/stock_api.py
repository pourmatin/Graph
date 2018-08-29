# coding=utf-8
"""
PAT - the name of the current project.
stock_api.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
7 / 31 / 18 - the current system date.
6: 03 PM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""
from iexfinance import Stock
from iex import stock


class Iex:
    """
    Factory class to create the right API
    """
    def __init__(self, api_name, ticker):
        self._api = api_name
        self._ticker = ticker

    def get_api(self):
        """
        creates the right api based on the name
        :return: Api object
        """
        if self._api == 'iex':
            return stock(self._ticker)
        elif self._api == 'iexfinance':
            return Stock(self._ticker)

    def volume(self):
        """
        get the latest volume
        :return: double
        """
        if self._api == 'iex':
            return self.get_api().quote().get('latestVolume')
        elif self._api == 'iexfinance':
            return float(self.get_api().get_volume())

    def price(self):
        """
        get the latest price
        :return: int
        """
        if self._api == 'iex':
            return self.get_api().price()
        elif self._api == 'iexfinance':
            return self.get_api().get_price()
