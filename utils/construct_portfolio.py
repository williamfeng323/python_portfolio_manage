import numpy
import pandas
import cvxopt
from cvxopt import solvers
import warnings


def markowitz_portfolio(cov_mat,exp_returns, target_return,
                        allow_short=False, market_neutral=False):
    if market_neutral and not allow_short:
        warnings.warn("A market neutral portfolio implies shorting")
        allow_short = True

    n = len(cov_mat)
    P = cvxopt.matrix(cov_mat.values)
    q = cvxopt.matrix(0.0, (n, 1))
    # Constraints Gx <= h
    if not allow_short:
        # exp_returns*x >= target_return and x >= 0
        G = cvxopt.matrix(numpy.vstack((-exp_returns.values, -numpy.identity(n))))
        h = cvxopt.matrix(numpy.vstack((-target_return, +numpy.zeros((n, 1)))))
    else:
        # exp_returns*x >= target_return
        G = cvxopt.matrix(-exp_returns.values).T
        h = cvxopt.matrix(-target_return)

    # Constraints Ax = b
    # sum(x) = 1
    A = cvxopt.matrix(1.0, (1, n))

    if not market_neutral:
        b = cvxopt.matrix(1.0)
    else:
        b = cvxopt.matrix(0.0)
    # Solve
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    # Put weights into a labeled series
    weights = pandas.Series(sol['x'], index=cov_mat.index)
    return weights


def tangency_portfolio(cov_mat, exp_returns, allow_short=False):
    n = len(cov_mat)

    P = cvxopt.matrix(cov_mat.values)
    q = cvxopt.matrix(0.0, (n, 1))

    # Constraints Gx <= h
    if not allow_short:
        # exp_returns*x >= 1 and x >= 0
        G = cvxopt.matrix(numpy.vstack((-exp_returns.values, -numpy.identity(n))))
        h = cvxopt.matrix(numpy.vstack((-1.0, numpy.zeros((n, 1)))))
    else:
        # exp_returns*x >= 1
        G = cvxopt.matrix(-exp_returns.values).T
        h = cvxopt.matrix(-1.0)

    # Solve
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    # Put weights into a labeled series
    weights = pandas.Series(sol['x'], index=cov_mat.index)

    # Rescale weights, so that sum(weights) = 1
    weights /= weights.sum()
    return weights
