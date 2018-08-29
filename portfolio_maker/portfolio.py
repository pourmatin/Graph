# coding=utf-8
"""
PAT - the name of the current project.
portfolio.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
6 / 17 / 18 - the current system date.
10: 07 AM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""
import pandas
import datetime


class Portfolio:
    """
    Portfolio is a singleton class
    """
    def __init__(self):
        columns = ['Date', 'Ticker', 'Long', 'Size', 'Avg. Bug Price', 'Avg. Sell Price', 'Return']
        self._portfolio = pandas.DataFrame(columns=columns)

    def add(self, event):
        """
        adds the closed event to the portfolio

        :param event: the event object
        :return: the current return value
        """
        self._portfolio.append([datetime.date.today(),
                                event.instrument.ticker,
                                event.long,
                                event.size(),
                                event.avg_price('buy'),
                                event.avg_price('sell'),
                                event.return_rate()])
        return self.total_return()

    def total_return(self):
        """
        the total return of the portfolio

        :return double: dollar amount of the return
        """
        total = 0
        for event in self._portfolio.iterrows():
            buy = event['Avg. buy Price']
            sell = event['Avg. Sell Price']
            size = event['Size']
            change = (sell - buy) * size
            total += change if event['Long'] else - change
        return total

    @property
    def portfolio(self):
        """
        portfolio as dataFrame

        :return: portfolio
        """
        return self._portfolio

    def export_to_csv(self, path='./Results', filename='protfolio'):
        """
        export the portfolio to an excel file
        :return:
        """
        writer = pandas.ExcelWriter(path + '/' + filename + '.xlsx', engine='xlsxwriter')
        self._portfolio.to_excel(writer)
        writer.save()
