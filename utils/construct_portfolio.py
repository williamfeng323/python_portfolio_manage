import numpy
import pandas
import scipy.optimize as s_optimize
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.integrate import quad


def describe_portfolio(weights, returns, value=1e6, confident_interval=.99):
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


def calculate_omega(returns, weights, target):
    pf_returns = numpy.dot(returns, weights)
    kernel = gaussian_kde(pf_returns)

    def kernel_cdf(r):
        p = quad(kernel.evaluate, -numpy.inf, r)
        return p[0]

    def modified_kcdf(r):
        return 1.0-quad(kernel.evaluate, -numpy.inf, r)[0]

    return quad(modified_kcdf, target, numpy.inf)[0]/quad(kernel_cdf, -numpy.inf, target)[0]


def maximize_omega(returns, target):
    def omega(weights):
        pf_returns = numpy.dot(returns, weights)
        kernel = gaussian_kde(pf_returns)

        def kernel_cdf(r):
            p = quad(kernel.evaluate, -numpy.inf, r)
            return p[0]

        def modified_kcdf(r):
            return 1.0-quad(kernel.evaluate, -numpy.inf, r)[0]

        return -1.0*(quad(modified_kcdf, target, .42)[0]/quad(kernel_cdf, -numpy.inf, target)[0])

    numberOfStocks = returns.columns.size
    bounds = [(0., 1.) for i in numpy.arange(numberOfStocks)]
    weights = numpy.ones(returns.columns.size) / returns.columns.size
    constraint = ({
        'type': 'eq',
        'fun': lambda weights: numpy.sum(weights) - 1
    })
    results = s_optimize.minimize(omega, weights, method='SLSQP', constraints=constraint, bounds=bounds)
    return ['%.8f' % elem for elem in numpy.round(results.x, 4)]
