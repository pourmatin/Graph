# coding=utf-8
"""
PAT - the name of the current project.
bigquery.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
7 / 27 / 18 - the current system date.
9: 14 AM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""
import logging
import pandas
import threading
from google.cloud import bigquery
from google.api_core import exceptions
from config import LIVE_SCHEMA

# logging.basicConfig(filename='bigQuery.log', level=logging.ERROR)

class GoogleQuery:
    """
    class for read and write data
    """
    client = bigquery.Client()

    def __init__(self, ticker, dataset_id, table_id=None):
        """
        :param str dataset_id: name of the new dataset
        """
        self._ticker = ticker
        table_id = table_id or 'live_' + ticker
        self.dataset_id = self.read_or_new_dataset(dataset_id).dataset_id
        self.table_id = self.creat_table(table_id, LIVE_SCHEMA)

    @staticmethod
    def read_or_new_dataset(dataset_id):
        """
        reads the dataset or creates a new dataset in the project
        :param str dataset_id: name of the new dataset
        :return: dataset object
        """
        dataset_ids = [dataset.dataset_id for dataset in GoogleQuery.client.list_datasets()]
        if dataset_id in dataset_ids:
            return GoogleQuery.client.get_dataset(GoogleQuery.client.dataset('my_dataset'))
        elif threading.current_thread().name == 't0':
            dataset_ref = GoogleQuery.client.dataset(dataset_id)
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = 'US'
            dataset = GoogleQuery.client.create_dataset(dataset)
            return dataset

    def creat_table(self, table_id, schema):
        """
        creats table in the specified dataset with specified schema
        :param table_id:
        :param list schema: list of tuples with field name, type and mode e.g.[('full_name', 'STRING', mode='REQUIRED')]
        :return: table object
        """
        dataset_ref = GoogleQuery.client.dataset(self.dataset_id)
        table_ids = [table.table_id for table in GoogleQuery.client.list_tables(dataset_ref)]
        if table_id in table_ids:
            # table_ref = GoogleQuery.client.dataset('my_dataset').table('my_table')
            return table_id
        else:
            schema = [bigquery.SchemaField(*args) for args in schema]
            table_ref = dataset_ref.table(table_id)
            table = bigquery.Table(table_ref, schema=schema)
            GoogleQuery.client.create_table(table)  # API request
            table_ids.append(table_id)
            logging.info('Table {0} was created successfully'.format(table_id))
        return table_id

    def query(self, start=None, last=None):
        """
        method to query from Google's BigQuery
        :param datetime start: the start time of the table
        :param int last: the last N number of rows
        :return: dataframe
        """
        table = 'live_' + self._ticker
        if start:
            query = "SELECT * FROM {dataset}.{table} WHERE Time > '{limit}'" .format(dataset=self.dataset_id,
                                                                               table=table,
                                                                               limit=start.strftime('%Y-%m-%d'))
        elif last:
            query = "SELECT * FROM {dataset}.{table} ORDER BY Time DESC LIMIT {limit}" .format(dataset=self.dataset_id,
                                                                               table=table,
                                                                               limit=last)
        else:
            raise ValueError('Either start or last must be defined!')
        query_job = GoogleQuery.client.query(query, location='US')
        rows = [row.values() for row in query_job]
        result = pandas.DataFrame(data=rows, columns=['Time', 'Price', 'Volume'])
        result = result.set_index('Time')
        return result

    def write(self, rows):
        """
        appends a list of rows into a table
        :param list rows: list of tuples. e.g. [(u'Phred Phlyntstone', 32)]
        :return: list of errors, if any
        """
        table_ref = GoogleQuery.client.dataset(self.dataset_id).table(self.table_id)
        table = GoogleQuery.client.get_table(table_ref)
        try:
            errors = GoogleQuery.client.insert_rows(table, rows)  # API request
        except exceptions.BadRequest as e:
            errors = [e]
        if not errors:
            logging.info("Wrote {ticker} today's price in {table}".format(ticker=self._ticker, table=self.table_id))
        else:
            print(errors[0])
        return errors
