# coding=utf-8
"""
PAT - the name of the current project.
main_subscriber.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
8 / 8 / 18 - the current system date.
9: 14 AM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""

PROJECT_ID = 'inbound-trilogy-211314'
OHLC_SCHEMA = [('Time', 'DATETIME', 'REQUIRED'),
              ('Ticker', 'STRING', 'REQUIRED'),
              ('Open', 'FLOAT', 'REQUIRED'),
              ('High', 'FLOAT', 'REQUIRED'),
              ('Low', 'FLOAT', 'REQUIRED'),
              ('Close', 'FLOAT', 'REQUIRED'),
              ('Volume', 'INTEGER', 'REQUIRED')]

LIVE_SCHEMA = [('Time', 'DATETIME', 'REQUIRED'),
              # ('Ticker', 'STRING', 'REQUIRED'),
              ('Price', 'FLOAT', 'REQUIRED'),
               ('Volume', 'INTEGER', 'REQUIRED')]

json_file = '/Users/Hossein/Documents/MyCodes/Graph/inbound-trilogy-211314-e95d1d432e4e.json'
TICKERS = ['MMM',
           'AXP',
           'AAPL',
           'BA',
           'BAC',
           'CAT',
           'CVX',
           'CSCO',
           'KO',
           'DWDP',
           'XOM',
           'GS',
           'HD',
           'IBM',
           'INTC',
           'JNJ',
           'JPM',
           'MCD',
           'MRK',
           'MSFT',
           'NKE',
           'PFE',
           'PG',
           'TRV',
           'UNH',
           'UTX',
           'VZ',
           'V',
           'WMT',
           'WBA',
           'DIS'
           ]