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

from ttp_model.dataset import Labels, Datasets, DataSet
from price_fetcher.bigquery import GoogleQuery

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed

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
    print(ticker)
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
def cluster(data):
    """

    :param data:
    :return:
    """
    label = data.Label[0]
    start = data.index[0]
    end = None
    result = pd.DataFrame()
    # num_cols = {col: 'ohlc' for col in data.columns if col.startswith('F') or col.startswith('dF')}

    for row in data.itertuples(index=True):
        if label != row.Label:
            df = data.loc[start:end]
            if not df.empty:
                duration = str((end - start).seconds) + 'S'
                cluster1 = df.resample(rule=duration).mean()[:1]
                # cluster2 = df.resample(rule=duration).agg(num_cols)[:1]
                # cluster2.columns = ['_'.join(col) for col in cluster2.columns]
                clustered = cluster1
                clustered['Ticker'] = row.Ticker
                clustered['Volume'] = row.Volume
                result = result.append(clustered)
            label = row.Label
            start = row.Index
        else:
            end = row.Index
    return result

@timeit
def classify(data, period, frwd, pos_limit, neg_limit):
    """
    the classifier method to get the labels for the training data
    :param data: dataframe
    :param period: the period for the moving average on which we calculate the limits
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
        subset['min'] = subset.MA[(subset.MA.shift(1) > subset.MA) & (subset.MA.shift(-1) > subset.MA)]
        subset['max'] = subset.MA[(subset.MA.shift(1) < subset.MA) & (subset.MA.shift(-1) < subset.MA)]
        p0 = subset.MA.iloc[0]
        p_up = (pos_limit + 1.) * p0
        p_down = (neg_limit + 1.) * p0
        for row in subset.iterrows():
            if ~subset.isnull().loc[row[0], 'max']:
                if row[1]['max'] > p_up:
                    return Labels.BUY
                else:
                    return Labels.HOLD
            elif ~subset.isnull().loc[row[0], 'min']:
                if row[1]['min'] < p_down:
                    return Labels.SELL
                else:
                    return Labels.HOLD
        return Labels.HOLD

    temp['MA'] = moving_average(temp, period=period, ma_type='sma', field='Price')
    return temp.apply(_set_label, axis=1)

@timeit
def add_feature(data):
    """
    creates the features
    :param data:
    :return:
    """
    # ema_name = 'Fema' + str(period // 60)
    # sigma_name = 'Fsigma' + str(period // 60)
    # n_name = 'Fn'
    rsi_name = 'FRSI' + str(36)
    cols = [rsi_name]

    # data[ema_name] = self.moving_average(data, period=period, ma_type='ema', field='Change')
    # data[sigma_name] = self.moving_standard_deviation(data, period=period, ma_type='ema', field='Change')
    # data[n_name] = data.apply(lambda x: (x['Change'] - x[ema_name]) / x[sigma_name] if x[sigma_name] != 0 else 0
    #                           , axis=1)
    data[rsi_name] = relative_strength_index(data, 36)
    for col in cols:
        data['d' + col] = data[col].diff()
        # data[col + '_rate'] = data.apply()
    return data

@timeit
def build(tickers, period, frwd=1200, pos_limit=.003, neg_limit=-.001, path=None):
    """
    The main method to create the database
    :return: dataframe of features and labels
    """
    if path:
        data = pd.read_excel(path)
    else:
        data = pd.DataFrame(columns=['Price', 'Volume', 'Label'])
        for ticker in tickers:
            query = get_raw_data(ticker)
            if query.empty:
                continue
            _data = process_query(query)
            _data = add_feature(_data)
            _data['Label'] = classify(_data, period=period, frwd=frwd, pos_limit=pos_limit, neg_limit=neg_limit)
            freq = (_data.index[1] - _data.index[0]).seconds
            _data = _data[period:-frwd//freq]
            _data = cluster(_data)
            data = data.append(_data)
        writer = pd.ExcelWriter('./data.xlsx', engine='xlsxwriter')
        data.to_excel(writer)
        writer.save()

    # data = pd.DataFrame(data=numpy.random.normal((0, 10, 100), (.1, 1, 5), (10000, 3)),
    #                     index=pd.date_range(start=datetime.datetime.now(), periods=10000, freq='10S'),
    #                     columns=['F1', 'F2', 'F3'])
    # data['Label'] = data.apply(lambda x: 1 if x.F1 * 3 + 5 * x.F2 - .5 * x.F3 > 0 else 0, axis=1)
    # tr_ind = data.sample(frac=self._train_ratio).index
    tr_ind = data[data.index.date >= datetime.date(2018, 9, 24)].index
    y_tr = data[data.index.isin(tr_ind)]['Label'].values.astype(int)
    y_test = data[~data.index.isin(tr_ind)]['Label'].values.astype(int)
    y_ts = y_test[numpy.where(y_test == Labels.BUY)]
    features = [col for col in data.columns if col.startswith('F') or col.startswith('dF')]
    data = data[features]
    x_tr = data[data.index.isin(tr_ind)].values
    x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1, 1)
    x_ts = data[~data.index.isin(tr_ind)].values
    x_ts = x_ts.reshape(x_ts.shape[0], x_ts.shape[1], 1, 1)
    x_ts = x_ts[numpy.where(y_test == Labels.BUY)]
    y_tr = numpy.reshape(y_tr, -1)
    y_ts = numpy.reshape(y_ts, -1)
    dataset = Datasets(train=DataSet(x_tr, y_tr, one_hot=True, hm_classes=2),
                       test=DataSet(x_ts, y_ts, one_hot=True, hm_classes=2)
                       )
    return dataset
