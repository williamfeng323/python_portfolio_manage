import numpy
import pandas
# import cvxopt
# from cvxopt import solvers
import scipy.optimize as s_optimize
from scipy.stats import norm
import warnings


def describe_portfolio(weights, returns, value, confident_interval):
    pf_mean = numpy.dot(weights.T, returns.mean())
    pf_risk = numpy.sqrt(numpy.dot(weights.T, numpy.dot(returns.cov(), weights)))
    pf_var = calculate_var(weights, returns, value, confident_interval)
    return pf_mean, pf_risk, pf_var


def calculate_var(weights, returns, value=1e6, confident_interval=.01):
    if returns.empty:
        raise ValueError('covariance matrix was not defined')
    else:
        if not isinstance(returns, pandas.DataFrame):
            raise ValueError("Covariance matrix is not a DataFrame")
    if value not in locals():
        warnings.warn('portfolio initial value is not specified, 1e6 will be used as initial value')
    if confident_interval not in locals():
        warnings.warn('confident_interval initial value is not specified, .01 will be used as confident_interval')
    cov_mat = returns.cov()
    portfolio_mean = numpy.dot(weights.T, returns.mean())
    return (portfolio_mean + norm.ppf(confident_interval) * numpy.sqrt(numpy.dot(weights.T, numpy.dot(cov_mat, weights)))) * \
        numpy.sqrt(numpy.square(value))


def minimize_var(returns, confident_interval=.01, value=1e6):

    cov_mat = returns.cov()

    def var(weights):
        portfolio_mean = numpy.dot(weights.T, returns.mean())
        return numpy.absolute((portfolio_mean + norm.ppf(confident_interval) * numpy.sqrt(numpy.dot(weights.T,
                numpy.dot(cov_mat, weights)))) * numpy.sqrt(numpy.square(value)))

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
