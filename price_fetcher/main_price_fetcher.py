# coding=utf-8
"""
PAT - the name of the current project.
main_subscriber.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
7 / 27 / 18 - the current system date.
9: 14 AM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""

from observer import PriceFetcher
from publisher import delete_all_topics
from config import PROJECT_ID, TICKERS
import threading
import requests
import os


def get_tickers(filename=None, extension='txt'):
    """
    find the tickers from the '.txt' files in the current directory
    :param list filename: name of the files
    :param extension: the file type
    :return: list of tickers
    """
    filename = filename or [file for file in os.listdir('./source/') if file.split('.')[-1] == extension]
    tickers = []
    for file in filename:
        f = open('./source/' + file)
        lines = f.read().split('\n')[1:]
        tick = [line.split(',')[0] for line in lines]
        tick = [t for t in filter(None, tick)]
        tickers.extend(tick)
        f.close()
    tickers = list(set(tickers))
    tickers = [t for t in filter(lambda x: '.' not in x and '^' not in x, tickers)]
    tickers = [t.strip() for t in tickers]
    return [t for t in filter(None, tickers)]


def send_simple_message():
    """
    the email sender
    :return: post request
    """
    return requests.post(
        "https://api.mailgun.net/v3/YOUR_DOMAIN_NAME/messages",
        auth=("api", ''),
        data={"from": "Excited User <mailgun@sandbox1619118a9c7f450ca3a018dec9363672.mailgun.org>",
              "to": ["hpourmatin@gmail.com"],
              "subject": "Hello",
              "text": "Testing some Mailgun awesomness!"})


def main():
    """
    The main function to start the price fetcher
    """
    # send_simple_message()
    nthreads = 10
    tickers = TICKERS  # get_tickers(filename=['SPX.csv'])
    delete_all_topics(PROJECT_ID)
    ntickers = int(len(tickers)/nthreads)
    for ithread in range(nthreads):
        ilist = tickers[ithread * ntickers: (ithread + 1) * ntickers]
        scanner = PriceFetcher(tickers=ilist, topic='live_publisher_')
        thread = threading.Thread(target=scanner.engine, name='t' + str(ithread))
        thread.start()


if __name__ == '__main__':
    main()
