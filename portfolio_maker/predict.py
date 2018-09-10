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
import tensorflow as tf
import pandas as pd
import datetime
from price_fetcher.bigquery import GoogleQuery

chunck_size = 72
n_chuncks = 10

sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('../ttp_model/my_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('../ttp_model'))

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

def get_raw_data(ticker):
    """
    gets the data for the list of tickers
    :return: dataframe
    """
    google = GoogleQuery(ticker, dataset_id='my_dataset')
    query = google.query(last=3600)
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
        result = result.append(df)
    result = result.sort_index()
    return result

graph = tf.get_default_graph()
xx = graph.get_tensor_by_name("input:0")
output = graph.get_tensor_by_name("rnn_model:0")
prediction = tf.argmax(output, 1)
feed_dict = {xx: data.reshape((-1, n_chuncks, chunck_size))}


decition = sess.run(prediction, feed_dict)
