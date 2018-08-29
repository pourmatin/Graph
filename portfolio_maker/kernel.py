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

import pandas
import numpy
import math


def kernel_factory(kernel_name):
    """
    factory for kernel functions
    :param kernel_name: nameof the kerner
    :return: the corresponding function
    """
    if kernel_name.lower() == 'bollinger':
        return bollinger
    else:
        raise Exception('{0} is not a known kernel type!'.format(kernel_name))


class Strategy:
    """
    base class for the trading strategy
    """

    def __init__(self, kernel, rule=None):
        self._rule = rule or '1M'
        self._kernel = kernel

    @property
    def rule(self):
        """
        the sampling frequency
        :return: str
        """
        return self._rule

    @rule.setter
    def rule(self, value):
        self._rule = value

    def moving_average(self, instrument, period, matype=None, field=None):
        """
        Method to calculate the moving average 
        :param instrument: the instrument object
        :param int period: the window of rolling average
        :param str matype: type of moving average, either simple, 'sma' or exponensial, 'ema'
        :param str field: which field to use for the moving average. default is 'Close'
        :return:
        """
        if matype.lower() == 'sma':
            profile = instrument.snapshot(self.rule)[field].rolling(period, min_periods=0).mean()
        elif matype.lower() == 'ema':
            profile = instrument.snapshot(self.rule)[field].ewm(span=period).mean()
        else:
            raise Exception('{0} is not a known moving average type!'.format(matype))
        return profile

    def moving_standard_deviation(self, instrument, period, matype=None, field=None):
        """
        Method to calculate the moving standard deviation
        :param instrument: the instrument object
        :param int period: the window of rolling average
        :param str matype: type of moving average, either simple, 'sma' or exponensial, 'ema'
        :param str field: which field to use for the moving average. default is 'Close'
        :return:
        """
        field = field or 'Close'
        if matype.lower() == 'sma':
            profile = pandas.rolling_std(instrument.snapshot(self.rule)[field], period, min_periods=0)
        elif matype.lower() == 'ema':
            mean = self.moving_average(instrument, period, matype=matype, field=field)
            mean_sq = mean * mean
            sq_field = instrument.snapshot(self.rule)[field] * instrument.snapshot(self.rule)[field]
            sq_mean = sq_field.ewm(span=period).mean()
            profile = numpy.sqrt(sq_mean - mean_sq)
        else:
            raise Exception('{0} is not a known moving average type!'.format(matype))
        return profile

    def score(self, instrument):
        """
        The instrument score will be from 0 to 100. The higher the score
        the more likely the instrument's price will increace
        :param instrument: the instrument object
        :return: double
        """
        func = kernel_factory(self._kernel)
        return func(self, instrument)


def bollinger(strategy, instrument, period=20, matype=None, field=None):
    """
    The Bollinger band width kernel
    :param Strategy strategy: the instance of the Strategy class
    :param Instrument instrument: the instrument object
    :param int period: the window of rolling average
    :param str matype: type of moving average, either simple, 'sma' or exponensial, 'ema'
    :param str field: which field to use for the moving average. default is 'Close'
    :return double: the score
    """
    matype = matype or 'ema'
    field = field or 'Close'
    profile = instrument.snapshot(strategy.rule)[field].tail()
    profile['nu'] = strategy.moving_average(instrument, period, matype=matype, field=field).tail()
    profile['Sigma'] = strategy.moving_standard_deviation(instrument, period, matype=matype, field=field).tail()
    profile[['d' + col for col in profile.columns]] = profile.diff()

    def _score1(x):
        return (-1.25 * x * x * x * x + 22.25 * x * x + 1.) / 100.

    def _score2(x):
        return -math.atan(x) / 3 + .5

    price = profile.get('Price')[-1]
    sigma = profile.get('Sigma')[-1]
    nu = profile.get('nu')[-1]
    n = (price - nu) / sigma
    s1 = _score1(n)
    d_price = profile.get('dPrice')[-1]
    d_sigma = profile.get('dSigma')[-1]
    m = d_price / d_sigma
    s2 = _score2(m)
    return s1 * s2
