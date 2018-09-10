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

from regression import Regression
from price_fetcher.bigquery import GoogleQuery


def main():
    reg = Regression()
    google = GoogleQuery(ticker='AAPL', dataset_id='my_dataset', table_id='live_AAPL')
    reg.fit()


if __name__ == '__main__':
    main()
