import os
import pandas
import sys
import numpy

import utils.construct_portfolio as cp


def loadData(file_name):
    file_path = os.path.join('./data',file_name)
    returns = pandas.read_csv(file_path, index_col=0)
    cov_mat = returns.cov()
    avg_returns = returns.mean()
    return returns, cov_mat, avg_returns


def main():
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = 'stockReturns.csv'
        returns, cov_mat, avg_returns = loadData(file_name)

    value = 1e6  # Investing one million in the portfolio
    confident_interval = 0.99
    variance = numpy.diag(cov_mat)
    standard_deviation = numpy.sqrt(variance)

    resolve = pandas.to_numeric(pandas.Series(cp.minimize_var(returns, cov_mat, confident_interval, value)))

if __name__ == '__main__':
    main()
