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
import copy
import time
import logging
import os
import numpy as np
from ttp_model.dataset import Labels, Datasets, DataSet
from price_fetcher.bigquery import GoogleQuery
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from ttp_model.dataset import Labels, Datasets, DataSet
from price_fetcher.bigquery import GoogleQuery
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


logging.basicConfig(filename='info.log', level=logging.INFO)


def timeit(method):
    """
    decorator method to time the functions
    :param method: the method to time
    :return:
    """
    def _timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.info('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return _timed


def get_raw_data(ticker, start_date=None):
    """
    gets the data for the list of tickers
    :return: dataframe
    """
    start_date = start_date or datetime.date(2018, 1, 1)
    google = GoogleQuery(ticker, dataset_id='my_dataset')
    query = google.query(start=start_date)
    query = query.sort_index()
    return query


def process_query(query, freq=10):
    """

    :param query:
    :param freq:
    :return:
    """
    dates = list(set(query.index.date))
    result = pd.DataFrame(columns=['Price', 'Volume'])
    for date in dates:
        _start = query[query.index.date == date].index[0]
        _end = query[query.index.date == date].index[-1]
        start = max([datetime.datetime(date.year, date.month, date.day, 8, 30), _start])
        end = min([datetime.datetime(date.year, date.month, date.day, 15, 0), _end])
        df = pd.DataFrame(index=pd.date_range(start=start, end=end, freq=str(freq) + 'S'), columns=['Price', 'Volume'])
        df.index.name = 'Time'
        df = pd.merge_asof(df, query.sort_index(), left_index=True, right_index=True)
        df = df.rename(columns={'Price_y': 'Price', 'Volume_y': 'Volume'})
        df = df.drop([col for col in df.columns if col.endswith('_x')], axis=1)
        if not df.empty:
            p0 = df.iloc[0].Price
            df['Change'] = df.apply(lambda x: (x.Price - p0) / p0, axis=1)
            result = result.append(df)
    result = result.sort_index()
    return result


def moving_average(data, period, ma_type=None, field=None):
    """
    gets the moving average to the data frame
    :param data: the dataframe
    :param int period: the window of rolling average
    :param str ma_type: type of moving average, either simple, 'sma' or exponensial, 'ema'
    :param str field: the field on which we perform the moving average
    :return: a dataframe with the moving average
    """
    ma_type = ma_type or 'sma'
    #     freq = (data.index[1] - data.index[0]).seconds
    if field:
        data = data[field]
    if ma_type.lower() == 'sma':
        profile = data.rolling(period, min_periods=0).mean()
    elif ma_type.lower() == 'ema':
        profile = data.ewm(span=period).mean()
    else:
        raise ValueError('{0} is not a known moving average type!'.format(ma_type))
    return profile


def moving_standard_deviation(data, period, ma_type=None, field=None):
    """
    Method to calculate the moving standard deviation
    :param data: the dataframe
    :param int period: the window of rolling average
    :param str ma_type: type of moving average, either simple, 'sma' or exponensial, 'ema'
    :param field:
    :return:
    """
    field = field or 'Price'
    freq = (data.index[1] - data.index[0]).seconds
    if ma_type.lower() == 'sma':
        profile = pd.rolling_std(data[field], period, min_periods=0)
    elif ma_type.lower() == 'ema':
        mean = moving_average(data, period, ma_type=ma_type, field=field)
        mean_sq = mean * mean
        sq_field = data[field] * data[field]
        sq_mean = moving_average(sq_field, period, ma_type=ma_type)
        profile = numpy.sqrt(sq_mean - mean_sq)
        profile = profile.fillna(0)
    else:
        raise ValueError('{0} is not a known moving average type!'.format(ma_type))
    return profile


def relative_strength_index(df, period, field=None):
    """Calculate Relative Strength Index(RSI) for given data.

    :param df: pandas.DataFrame
    :param period:
    :param field:
    :return: pandas.DataFrame
    """
    field = field or 'Price'
    temp = copy.deepcopy(df)
    temp['Change'] = temp[field].diff()
    temp['Gain'] = temp['Change'].apply(lambda x: max([x, 0]))
    temp['Loss'] = temp['Change'].apply(lambda x: abs(min([x, 0])))
    temp['AG'] = temp['Gain'].rolling(window=period).mean()
    temp['AL'] = temp['Loss'].rolling(window=period).mean()
    return temp.apply(lambda x: 100 - 100 / (1 + x.AG / x.AL) if x.AL != 0 else 0, axis=1)


def classify(data, frwd, pos_limit=.001, neg_limit=-.001):
    """
    the classifier method to get the labels for the training data
    :param data: dataframe
    :param frwd: the forward limit to look for extremums
    :param pos_limit:
    :param neg_limit:
    :return:
    """
    temp = copy.deepcopy(data)

    temp['change'] = temp['Price'].diff() / temp['Price'].shift()
    return temp['change'].apply(lambda x: 1 if x > pos_limit else 0)


def macd(data, field):
    ma12 = data[field].ewm(span=12).mean()
    ma26 = data[field].ewm(span=26).mean()
    return ma12 - ma26


@timeit
def build():
    """
    The main method to create the database
    :return: dataframe of features and labels
    """
    rootdir = '/Users/Hossein/Downloads/NYSE_2018/'
    query = pd.DataFrame()
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            query = query.append(pd.read_csv(subdir + file))
    query = query.rename(columns={'<close>': 'Price', '<ticker>': 'Ticker'})
    query['Time'] = query['<date>'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))
    dataset = query.set_index('Time')
    dataset = dataset[dataset['<vol>'] > 0]
    dataset['dummyindex'] = dataset.index
    dataset = dataset.sort_values(['Ticker', 'dummyindex'])
    dataset['Change'] = dataset['Price'].diff() / dataset['Price'].shift()
    dataset = dataset[abs(dataset['Change']) < .05]
    dataset['Label'] = dataset['Change'].apply(lambda x: 1 if x > .01 else 0)
    data = dataset.groupby('Ticker').max()['Price']
    dataset = dataset.merge(pd.DataFrame(data=data), left_on='Ticker', right_index=True)
    dataset['scaled'] = dataset.Price_x / dataset.Price_y
    dataset.drop(['<date>', '<high>', '<low>', '<open>', 'dummyindex', 'Price_x', 'Price_y'], inplace=True, axis=1)
    dataset['SMA60'] = dataset['scaled'].rolling(window=60, min_periods=0, center=False).mean()
    dataset['STD'] = dataset['scaled'].rolling(window=60, min_periods=0, center=False).std()
    dataset = dataset[dataset.STD > 0]
    dataset['N'] = (dataset['scaled'] - dataset['SMA60']) / dataset['STD']
    dataset['RSI'] = relative_strength_index(dataset, period=14, field='scaled')
    dataset['MACD'] = macd(dataset, 'scaled')
    dataset['Date'] = dataset.index.date
    dataset['Day'] = dataset['Date'].apply(lambda x: x.day)
    dataset['Month'] = dataset['Date'].apply(lambda x: x.month)
    dataset['Year'] = dataset['Date'].apply(lambda x: x.year)
    dataset = dataset.dropna()

    ticker_dummy = pd.get_dummies(dataset['Ticker'], prefix='ticker')
    data = dataset.join(ticker_dummy)
    le = preprocessing.LabelEncoder()
    le.fit(dataset['Ticker'].values)
    feature_list = ['N', 'RSI', 'MACD', 'Day', 'SMA60'] + ['ticker_' + i for i in le.classes_]
    data = data[feature_list + ['Label']]

    X_train, X_test, y_train, y_test = train_test_split(X, dataset[y], test_size=0.3, random_state=2)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {0}%'.format(logreg.score(X_test, y_test) * 100))
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)


class DataProcessor:
    def __init__(self, tickers, frwd=1200, start_date=None, path=None):
        """

        :param tickers:
        :param frwd:
        :param start_date:
        :param path:
        """
        self._tickers = tickers
        self._forward = frwd
        self._start = start_date
        self._path = path
        self.n_labels = 20

    def featurize(self, data):
        _, bins = np.histogram(data['Change'].values, bins=self.n_labels)
        data['F0'] = data['Change'].apply(lambda x: max([np.searchsorted(bins, x) - 1, 0]))
        for i in range(120):
            data['F' + str(i + 1)] = data['Change'].shift(i).apply(lambda x: max([np.searchsorted(bins, x) - 1, 0]))
        data['Label'] = data['Change'].shift(-1).apply(lambda x: max([np.searchsorted(bins, x) - 1, 0]))
        return data

    def build(self, period=120):
        """
        The main method to create the database

        :return:
        """
        if self._path:
            data = pd.read_excel(self._path)
        else:
            data = pd.DataFrame()
            for ticker in self._tickers:
                query = get_raw_data(ticker, self._start)
                if query.empty:
                    continue
                print('got the data')
                _data = process_query(query)
                print('processed the data')
                _data = self.featurize(_data)
                print('featurized the data')
                freq = (_data.index[1] - _data.index[0]).seconds
                _data = _data[period:-self._forward//(freq)]
                data = data.append(_data)
            writer = pd.ExcelWriter('./data.xlsx', engine='xlsxwriter')
            data.to_excel(writer)
            writer.save()

        tr_ind = data[data.index.date >= datetime.date(2018, 11, 1)].index
        y_tr = data[data.index.isin(tr_ind)]['Label'].values.astype(int)
        y_ts = data[~data.index.isin(tr_ind)]['Label'].values.astype(int)
        features = [col for col in data.columns if col.startswith('F') or col.startswith('dF')]
        changes = data['Change']
        data = data[features]
        x_tr = data[data.index.isin(tr_ind)].values
        x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1, 1)
        x_ts = data[~data.index.isin(tr_ind)].values
        x_ts = x_ts.reshape(x_ts.shape[0], x_ts.shape[1], 1, 1)
        y_tr = numpy.reshape(y_tr, -1)
        y_ts = numpy.reshape(y_ts, -1)
        dataset = Datasets(train=DataSet(x_tr, y_tr, one_hot=True, hm_classes=self.n_labels),
                           test=DataSet(x_ts, y_ts, one_hot=True, hm_classes=self.n_labels)
                           )
        _, bins = np.histogram(changes.values, bins=self.n_labels)
        return dataset, bins