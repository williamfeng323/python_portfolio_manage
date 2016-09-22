import numpy
import pandas
import scipy.optimize as s_optimize
from scipy.stats import norm


def describe_portfolio(weights, returns, value, confident_interval):
    pf_mean = numpy.dot(weights.T, returns.mean())
    pf_risk = numpy.sqrt(numpy.dot(weights.T, numpy.dot(returns.cov(), weights)))
    pf_var = calculate_var(weights, returns, value, confident_interval)
    return pf_mean, pf_risk, pf_var


def calculate_var(weights, returns, value=1e6, confident_interval=.99):
    if returns.empty:
        raise ValueError('covariance matrix was not defined')
    else:
        if not isinstance(returns, pandas.DataFrame):
            raise ValueError("Covariance matrix is not a DataFrame")
    cov_mat = returns.cov()
    portfolio_mean = numpy.dot(weights.T, returns.mean())
    sigma = numpy.sqrt(numpy.dot(weights.T, numpy.dot(cov_mat, weights)))
    # (mean + confident_interval * sigma) * portfolio_size
    return norm.interval(confident_interval, loc=portfolio_mean, scale=sigma/numpy.sqrt(len(returns.index)))[0] * \
        numpy.sqrt(numpy.square(value))


def minimize_var(returns, confident_interval=.01, value=1e6):

    cov_mat = returns.cov()

    def var(weights):
        portfolio_mean = numpy.dot(weights.T, returns.mean())
        sigma = numpy.sqrt(numpy.dot(weights.T, numpy.dot(cov_mat, weights)))
        # (mean + confident_interval * sigma) * portfolio_size
        return numpy.absolute(norm.interval(confident_interval, loc=portfolio_mean, scale=sigma / numpy.sqrt(len(returns.index)))[0] *
                              numpy.sqrt(numpy.square(value)))

    numberOfStocks = returns.columns.size
    bounds = [(0., 1.) for i in numpy.arange(numberOfStocks)]
    weights = numpy.ones(returns.columns.size) / returns.columns.size
    constraint = ({
        'type': 'eq',
        'fun': lambda weights: numpy.sum(weights)-1
    })
    results = s_optimize.minimize(var, weights, method='SLSQP', constraints=constraint, bounds=bounds)
    return ['%.8f' % elem for elem in numpy.round(results.x, 4)]

