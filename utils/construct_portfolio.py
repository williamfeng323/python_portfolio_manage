import numpy
import pandas
# import cvxopt
# from cvxopt import solvers
import scipy.optimize as s_optimize
from scipy.stats import norm
import warnings


def calculate_var(weights, cov_mat, value=1e6, confident_interval=.99):
    if cov_mat.empty:
        raise ValueError('covariance matrix was not defined')
    else:
        if not isinstance(cov_mat, pandas.DataFrame):
            raise ValueError("Covariance matrix is not a DataFrame")
    if value not in locals():
        warnings.warn('portfolio initial value is not specified, 1e6 will be used as initial value')
    if confident_interval not in locals():
        warnings.warn('confident_interval initial value is not specified, .99 will be used as confident_interval')

    return norm.ppf(confident_interval) * numpy.sqrt(numpy.square(value)) * (numpy.dot(weights.T, numpy.dot(cov_mat, weights)))


def minimize_var(returns, cov_mat, confident_interval=.99, value=1e6):
    def var(weights):
        return norm.ppf(confident_interval) * numpy.sqrt(numpy.square(value)) * \
               (numpy.dot(weights.T, numpy.dot(cov_mat, weights)))

    numberOfStocks = returns.columns.size
    bounds = [(0., 1.) for i in numpy.arange(numberOfStocks)]
    weights = numpy.ones(returns.columns.size) / returns.columns.size
    constraint = ({
        'type': 'eq',
        'fun': lambda weights: numpy.sum(weights)-1
    })
    results = s_optimize.minimize(var, weights, method='SLSQP', constraints=constraint, bounds=bounds)
    return ['%.8f' % elem for elem in numpy.round(results.x, 4)]

#
# def set_variables(init_returns, init_cov_mat, init_value, init_confident_interval):
#     returns = init_returns
#     cov_Mat = init_cov_mat
#     value = init_value
#     confident_interval = init_confident_interval
#     return True
