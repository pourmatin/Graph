# coding=utf-8
"""
Graph - the name of the current project.
instrument.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
6 / 15 / 18 - the current system date.
8: 03 AM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""

import numpy
import datetime
import pandas as pd
import collections
import functools
import copy

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from price_fetcher.bigquery import GoogleQuery


Datasets = collections.namedtuple('Datasets', ['train', 'test'])


class Labels:
    """
    Label deffinitions
    """
    BUY = 2
    HOLD = 1
    SELL = 0


class DataSet(object):
    """
    Container class for a dataset (deprecated).
    """
    
    def __init__(self, features, labels, one_hot=False, dtype=dtypes.float32, reshape=True, seed=None):
        """
        Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                'Invalid image dtype %r, expected uint8 or float32' % dtype)
        assert features.shape[0] == labels.shape[0], (
          'features.shape: %s labels.shape: %s' % (features.shape, labels.shape))
        self._num_examples = features.shape[0]
        
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert features.shape[3] == 1
        features = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
        if dtype == dtypes.float32:
            features = features.astype(numpy.float32)
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

        if one_hot:
            classes = list(set(labels))
            hm_classes = int(max(classes)) + 1
            hot_array = numpy.zeros((self._num_examples, hm_classes))
            hot_array[numpy.arange(self._num_examples), labels] = 1
            self._labels = hot_array
    
    @property
    def features(self):
        """
        getter method for features
        :return: features
        """
        return self._features
    
    @property
    def labels(self):
        """
        getter method for labels
        :return: labels
        """
        return self._labels
    
    @property
    def num_examples(self):
        """
        getter method for number of datasets
        :return:
        """
        return self._num_examples
    
    @property
    def epochs_completed(self):
        """
        getter method for the number of completed epochs
        :return:
        """
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._features = self.features[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate(
              (features_rest_part, features_new_part)), numpy.concatenate(
                  (labels_rest_part, labels_new_part))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end]


class AcquireData:
    """
    class to get the input data and find the features and labels
    """
    def __init__(self,
                 tickers=None,
                 start_date=None,
                 data_period=1200,
                 pos_limit=.003,
                 neg_limit=-.001,
                 forward_limit=1200,
                 backward_limit=600,
                 train_ratio=.7,
                 valid_ratio=.1):
        self.tickers = tickers
        self._start_date = start_date or datetime.date(2018, 1, 1)
        self._pos = pos_limit
        self._neg = neg_limit
        self._frwd = forward_limit
        self._bcwd = backward_limit
        self._train_ratio = train_ratio
        self._valid_ratio = valid_ratio
        self._freq = 10
        self.period = data_period

    @functools.lru_cache(maxsize=None)
    def get_raw_data(self, ticker):
        """
        gets the data for the list of tickers
        :return: dataframe
        """
        google = GoogleQuery(ticker, dataset_id='my_dataset')
        query = google.query(start=self._start_date)
        query = query.sort_index()
        if query.empty:
            return query
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
            df['Ticker'] = ticker
            result = result.append(df)
        result = result.sort_index()
        return result

    def moving_average(self, data, period=None, ma_type=None, field=None):
        """
        gets the moving average to the data frame
        :param data: the dataframe
        :param int period: the window of rolling average
        :param str ma_type: type of moving average, either simple, 'sma' or exponensial, 'ema'
        :param str field: the field on which we perform the moving average
        :return: a dataframe with the moving average
        """
        ma_type = ma_type or 'sma'
        period = period or self.period
        if field:
            data = data[field]
        if ma_type.lower() == 'sma':
            profile = data.rolling(int(period / self._freq), min_periods=0).mean()
        elif ma_type.lower() == 'ema':
            profile = data.ewm(span=int(period / self._freq)).mean()
        else:
            raise ValueError('{0} is not a known moving average type!'.format(ma_type))
        return profile

    def moving_standard_deviation(self, data, period=None, ma_type=None):
        """
        Method to calculate the moving standard deviation
        :param data: the dataframe
        :param int period: the window of rolling average
        :param str ma_type: type of moving average, either simple, 'sma' or exponensial, 'ema'
        :return:
        """
        period = period or self.period
        if ma_type.lower() == 'sma':
            profile = pd.rolling_std(data['Price'], period / self._freq, min_periods=0)
        elif ma_type.lower() == 'ema':
            mean = self.moving_average(data, period, ma_type=ma_type, field='Price')
            mean_sq = mean * mean
            sq_field = data['Price'] * data['Price']
            sq_mean = self.moving_average(sq_field, period, ma_type=ma_type)
            profile = numpy.sqrt(sq_mean - mean_sq)
            profile = profile.fillna(0)
        else:
            raise ValueError('{0} is not a known moving average type!'.format(ma_type))
        return profile

    def _get_crossing(self, subset):
        p0 = subset.iloc[0].Price
        p_up = (self._pos + 1.) * p0
        p_down = (self._neg + 1.) * p0
        subset = subset[(subset.MA > p_up) | (subset.MA < p_down)]
        if not subset.empty:
            return subset.iloc[0].MA

    def classify(self, data, period):
        """
        the classifier method to get the labels for the training data
        :param data: dataframe
        :param period: the period for the moving average on which we calculate the limits
        :return:
        """
        temp = copy.deepcopy(data)

        def _find_next_big_change(row):
            ts = (row.name - numpy.datetime64('1970-01-01T00:00:00Z')) / numpy.timedelta64(1, 's')
            start = datetime.datetime.utcfromtimestamp(ts)
            end = start + datetime.timedelta(seconds=self._frwd)
            subset = temp.loc[start:end]
            p0 = subset[subset.Ticker == row.Ticker].iloc[0].Price
            cross_value = self._get_crossing(subset)
            if cross_value:
                return Labels.BUY if cross_value > p0 else Labels.SELL
            else:
                return Labels.HOLD

        temp['MA'] = self.moving_average(temp, period=period, ma_type='sma', field='Price')
        return temp.apply(_find_next_big_change, axis=1)

    def add_feature(self, data, period):
        push_back = int(self._bcwd / self._freq)
        ema_name = 'Fema' + str(period / 60)
        sigma_name = 'Fsigma' + str(period / 60)
        n_name = 'Fn' + str(period / 60)
        temp = copy.deepcopy(data)

        def _flip(df):
            col = list(df.columns)
            col.reverse()
            return df[col]

        ema = temp[ema_name] = self.moving_average(data, period=period, ma_type='ema', field='Price')
        ema = ema.to_frame(ema_name)
        sigma = temp[sigma_name] = self.moving_standard_deviation(data, period=period, ma_type='ema')
        sigma = sigma.to_frame(sigma_name)
        n = temp.apply(lambda x: (x['Price'] - x[ema_name]) / x[sigma_name] if x[sigma_name] != 0 else 0, axis=1)
        n = n.to_frame(n_name)
        for i in range(1, push_back):
            ema[ema_name + '-' + str(i)] = ema[ema_name].shift(i)
            n[n_name + str(i)] = n[n_name].shift(i)
            sigma[sigma_name + str(i)] = sigma[sigma_name].shift(i)
        ema = _flip(ema)
        n = _flip(n)
        sigma = _flip(sigma)
        data = data.join(ema)
        data = data.join(sigma)
        data = data.join(n)
        return data

    def build(self):
        """
        The main method to create the database
        :return: dataframe of features and labels
        """
        data = pd.DataFrame(columns=['Price', 'Volume', 'Label'])
        for ticker in self.tickers:
            _data = self.get_raw_data(ticker)
            if _data.empty:
                continue
            for period in range(60, self.period + 1, 300):
                _data = self.add_feature(_data, period)
            _data['Label'] = self.classify(_data, period=60)
            _data = _data[self.period//self._freq:-self._frwd//self._freq]
            data = data.append(_data)
        # tr_ind = data.sample(frac=self._train_ratio).index
        tr_ind = data[data.index.date != datetime.date.today()]
        y_tr = data[data.index.isin(tr_ind)]['Label'].values.astype(int)
        y_test = data[~data.index.isin(tr_ind)]['Label'].values.astype(int)
        y_ts = y_test#[numpy.where(y_test == Labels.BUY)]
        features = [col for col in data.columns if col.startswith('F')]
        data = data[features]
        x_tr = data[data.index.isin(tr_ind)].values
        x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1, 1)
        x_ts = data[~data.index.isin(tr_ind)].values
        x_ts = x_ts.reshape(x_ts.shape[0], x_ts.shape[1], 1, 1)
        # x_ts = x_ts[numpy.where(y_test == Labels.BUY)]
        y_tr = numpy.reshape(y_tr, -1)
        y_ts = numpy.reshape(y_ts, -1)
        dataset = Datasets(train=DataSet(x_tr, y_tr, one_hot=True),
                           test=DataSet(x_ts, y_ts, one_hot=True))
        return dataset
