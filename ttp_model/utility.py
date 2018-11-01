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

from ttp_model.dataset import Labels, Datasets, DataSet
from price_fetcher.bigquery import GoogleQuery

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
    query['Ticker'] = ticker
    logging.info(ticker)
    return query


def process_query(query, freq=10):
    """

    :param query:
    :param freq:
    :return:
    """
    dates = list(set(query.index.date))
    ticker = query.iloc[0].Ticker
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
        df['Ticker'] = ticker
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
    freq = (data.index[1] - data.index[0]).seconds
    if field:
        data = data[field]
    if ma_type.lower() == 'sma':
        profile = data.rolling(int(period / freq), min_periods=0).mean()
    elif ma_type.lower() == 'ema':
        profile = data.ewm(span=int(period / freq)).mean()
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
        profile = pd.rolling_std(data[field], period / freq, min_periods=0)
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


@timeit
def cluster(data, resample, freq='min'):
    """
    method to cluster the similar labels together
    :param data:
    :param resample:
    :param freq:
    :return:
    """

    num_cols = {col: 'ohlc' for col in data.columns if col.startswith('F') or col.startswith('dF')}
    result = data.resample(rule=str(resample) + freq).agg(num_cols)
    result.columns = ['_'.join(col) for col in result.columns]
    result['Label'] = data['Label'].resample(rule=str(resample) + freq, how='mean').values.astype(int)
    start = datetime.time(8, 30)
    end = datetime.time(15, 0)
    result = result[(result.index.time > start) & (result.index.time < end)]
    return result.dropna(subset=['F0_open'])


@timeit
def classify(data, frwd, pos_limit, neg_limit):
    """
    the classifier method to get the labels for the training data
    :param data: dataframe
    :param frwd: the forward limit to look for extremums
    :param pos_limit:
    :param neg_limit:
    :return:
    """
    temp = copy.deepcopy(data)

    def _set_label(row):
        ts = (row.name - numpy.datetime64('1970-01-01T00:00:00Z')) / numpy.timedelta64(1, 's')
        start = datetime.datetime.utcfromtimestamp(ts)
        date = start.date()
        eod = datetime.datetime(date.year, date.month, date.day, 15, 0)
        end = start + datetime.timedelta(seconds=frwd)
        end = min([end, eod])
        subset = copy.deepcopy(temp.loc[start:end])
        p0 = subset.MA.iloc[0]
        p_up = (pos_limit + 1.) * p0
        p_down = (neg_limit + 1.) * p0
        high = subset.MA[(subset.MA.shift(1) > subset.MA) & (subset.MA.shift(-1) > subset.MA) & (subset.MA > p_up)]
        low = subset.MA[(subset.MA.shift(1) < subset.MA) & (subset.MA.shift(-1) < subset.MA) & (subset.MA < p_down)]
        if high.empty:
            return Labels.HOLD if low.empty else Labels.SELL
        elif low.empty:
            return Labels.HOLD if high.empty else Labels.BUY
        else:
            return Labels.BUY if high.index[0] > low.index[0] else Labels.SELL

    temp['MA'] = moving_average(temp, period=60, ma_type='sma', field='Price')
    return temp.apply(_set_label, axis=1)


@timeit
def add_feature(data):
    """
    creates the features
    :param data:
    :return:
    """
    # period = 1200
    # ema_name = 'Fema' + str(period // 60)
    # sigma_name = 'Fsigma' + str(period // 60)
    # n_name = 'Fn'
    rsi_name = 'FRSI' + str(36)
    # streak = 'Fstreak'
    nstreak = 10
    #
    # cols = ['diff' + str(i) for i in range(1, nstreak)]
    #
    # def _find_streak(row):
    #     for i, column in enumerate(cols[:-1]):
    #         if (row[cols[i]] - row[cols[i+1]]) * (row['Price'] - row[cols[0]]) <= 0:
    #             return int(column[4:])
    #     return int(cols[-1][4:])
    #
    # data[ema_name] = moving_average(data, period=period, ma_type='ema', field='Change')
    # data[sigma_name] = moving_standard_deviation(data, period=period, ma_type='ema', field='Change')
    # data[n_name] = data.apply(lambda x: (x['Change'] - x[ema_name]) / x[sigma_name] if x[sigma_name] != 0 else 0,
    #                           axis=1)
    data[rsi_name] = relative_strength_index(data, 36)
    temp = copy.deepcopy(data)
    for i in range(1, nstreak):
        temp['diff' + str(i)] = data['Price'].shift(i)

    # data[streak] = temp.apply(_find_streak, axis=1)
    # cols = [rsi_name, streak, ema_name, sigma_name, n_name]
    # for col in cols:
    #     data['d' + col] = data[col].diff()
    temp['MA'] = moving_average(data, period=240, ma_type='sma', field='Change')
    temp['REF'] = temp['MA'].shift(nstreak)
    for i in range(nstreak):
        data['F' + str(i)] = temp['MA'].shift(i) - temp['REF']
    return data


@timeit
def build(tickers, period, resample=5, frwd=1200, pos_limit=.003, neg_limit=-.001, start_date=None, path=None):
    """
    The main method to create the database
    :return: dataframe of features and labels
    """
    if path:
        data = pd.read_excel(path)
    else:
        data = pd.DataFrame(columns=['Price', 'Volume', 'Label'])
        for ticker in tickers:
            query = get_raw_data(ticker, start_date)
            if query.empty:
                continue
            _data = process_query(query)
            _data = add_feature(_data)
            _data['Label'] = classify(_data, frwd=frwd, pos_limit=pos_limit, neg_limit=neg_limit)
            freq = (_data.index[1] - _data.index[0]).seconds
            _data = _data[period:-frwd//(freq * resample)]
            # _data = cluster(_data, resample)
            data = _data.append(_data)
        writer = pd.ExcelWriter('./data.xlsx', engine='xlsxwriter')
        data.to_excel(writer)
        writer.save()

    holds = len(data)
    buys = len(data[data.Label == Labels.BUY])
    weights = data.apply(lambda x: .0001 if x.Label == Labels.HOLD else 1, axis=1)
    data = data.sample(frac=2. * float(buys)/float(holds), weights=weights)
    tr_ind = data[data.index.date >= datetime.date(2018, 10, 1)].index
    y_tr = data[data.index.isin(tr_ind)]['Label'].values.astype(int)
    y_ts = data[~data.index.isin(tr_ind)]['Label'].values.astype(int)
    features = [col for col in data.columns if col.startswith('F') or col.startswith('dF')]
    data = data[features]
    x_tr = data[data.index.isin(tr_ind)].values
    x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1, 1)
    x_ts = data[~data.index.isin(tr_ind)].values
    x_ts = x_ts.reshape(x_ts.shape[0], x_ts.shape[1], 1, 1)
    y_tr = numpy.reshape(y_tr, -1)
    y_ts = numpy.reshape(y_ts, -1)
    dataset = Datasets(train=DataSet(x_tr, y_tr, one_hot=True, hm_classes=2),
                       test=DataSet(x_ts, y_ts, one_hot=True, hm_classes=2)
                       )
    return dataset
