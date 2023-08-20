"""RT algorithm

'Nonparameteric Estimation in Renewal Processes'
Y. Vardi
The Annals of Statistics, 1962, Vol. 10, No. 3, 772-785
"""
import operator
import functools

import numpy
from scipy.optimize import fsolve
from tqdm.auto import tqdm


def r_k(t, x, y, z, w, p_old, k):
    _sum = 0
    for i in range(k):
        t1 = (y[i] + z[i]) / numpy.sum(p_old[i:])
        t2 = (t[k] - t[i] + 1)*w[i] / numpy.sum((t[i:]-t[i] + 1)*p_old[i:])
        _sum += (t1 + t2)
    return x[k] + p_old[k] * _sum


def _r_k(T, p_old, k):
    return r_k(T[:,0], T[:,1], T[:,2], T[:,3], T[:,4], p_old, k)


def R_k(T, p_old):
    fun = functools.partial(_r_k, T, p_old)
    return list(map(fun, range(T.shape[0])))


def _mu_eqn(rk, T, n_xyzw, mu):
    terms = []
    for k in range(T.shape[0]):
        terms.append(
            rk[k]*T[k,0]/((n_xyzw[0]+n_xyzw[2])*mu + (n_xyzw[1]+n_xyzw[3])*T[k,0])
        )
    return functools.reduce(operator.add, terms) - 1


def solve_mu(rk, T, n_xyzw, mu0=0):
    opt_fun = functools.partial(_mu_eqn, rk, T, n_xyzw)
    return fsolve(opt_fun, mu0)[0]


def _stepB(T, n_xyzw, p_old, mu0=0):
    rk = numpy.asarray(R_k(T, p_old))
    mu = solve_mu(rk, T, n_xyzw, mu0)
    return rk, mu


def _stepC(T, rk, mu, n_xyzw):
    pk = rk * mu / ((n_xyzw[0]+n_xyzw[2])*mu + (n_xyzw[1]+n_xyzw[3])*T[:,0])
    return pk


def rt_iter(T, n_xyzw, p_old, mu0=0):
    """does a single iteration of the modified RT algorithm
    """
    rk, mu = _stepB(T, n_xyzw, p_old, mu0)
    p_new = _stepC(T, rk, mu, n_xyzw)
    return p_new, mu


def rt_algorithm(T, p_old, M, n_iter, mu0=0, verbose=False):
    """the RT algorithm
    the input array, T, should have 5 columns like [t, x, y, z, w]
    (as in the 'Data' side of Table 1 in Vardi)
    
    p_old is initial guess for p
    mu0 is an initial guess for the intermediate, mu, parameter
    """
    delp = numpy.zeros((n_iter))
    T = numpy.vstack([T, [M, 0, 0, 0, 0]])
    n_xyzw = numpy.sum(T, axis=0)[1:]
    if verbose:
        iterator = tqdm(range(n_iter), total=n_iter)
    else:
        iterator = range(n_iter)
    for i in iterator:
        # do the iteration
        p_new, mu_new = rt_iter(T, n_xyzw, p_old, mu0)
        # compute change in parameters
        delp[i] = numpy.sqrt(numpy.sum(numpy.square(p_new-p_old)))
        # new is next iter's old
        p_old, mu0 = p_new.copy(), mu_new
    return p_new, delp


def initialize_p(T):
    """make initial guess for p as uniform distribution
    """
    p = numpy.ones((T.shape[0]+1,)) / (T.shape[0]+1)
    return p
