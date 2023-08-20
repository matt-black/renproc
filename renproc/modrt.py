"""Modified RT algorithm

'Nonparametric estimation and consistency for renewal processes'
Guoxing Soon, Michael Woodrofe
Journal of Statistical Planning and Inference, 53 (1996) 171-195
"""
import operator
import functools

import numpy
from scipy.optimize import fsolve
from tqdm.auto import tqdm


def r_k(t, x, y, z, w, p_old, v_old, k):
    _sum = 0
    for i in range(k):
        t1 = (y[i] + z[i]) / numpy.sum(p_old[i:])
        t2 = (t[k] - t[i])*w[i] / (numpy.sum((t[i:]-t[i])*p_old[i:]) + v_old)
        _sum += (t1 + t2)
    return x[k] + p_old[k] * _sum


def _r_k(T, p_old, v_old, k):
    return r_k(T[:,0], T[:,1], T[:,2], T[:,3], T[:,4], p_old, v_old, k)


def r_h1(t, w, p_old, v_old, h):
    out = 0
    for i in range(h):
        out += w[i] / (numpy.sum((t[i:]-t[i])*p_old[i:]) + v_old)
    return v_old * out


def _r_h1(T, p_old, v_old):
    return r_h1(T[:,0], T[:,-1], p_old, v_old, T.shape[0])


def R_k(T, p_old, v_old):
    fun = functools.partial(_r_k, T, p_old, v_old)
    return list(map(fun, range(T.shape[0])))


def _mu_eqn(rk, T, n_xyzw, p_old, v_old, rh1, mu):
    terms = []
    for k in range(T.shape[0]):
        terms.append(
            rk[k]*T[k,0]/((n_xyzw[0]+n_xyzw[2])*mu + (n_xyzw[1]+n_xyzw[3])*T[k,0])
        )
    lhs = 1 - rh1 / (n_xyzw[1]+n_xyzw[3])
    return functools.reduce(operator.add, terms) - lhs


def solve_mu(rk, T, n_xyzw, p_old, v_old, mu0=0):
    rh1 = _r_h1(T, p_old, v_old)
    opt_fun = functools.partial(_mu_eqn, rk, T, n_xyzw, p_old, v_old, rh1)
    return fsolve(opt_fun, mu0)[0]


def _stepB(T, n_xyzw, p_old, v_old, mu0=0):
    rk = numpy.asarray(R_k(T, p_old, v_old))
    mu = solve_mu(rk, T, n_xyzw, p_old, v_old, mu0)
    return rk, mu


def _stepC(T, rk, mu, n_xyzw, rh1):
    pk = rk * mu / ((n_xyzw[0]+n_xyzw[2])*mu + (n_xyzw[1]+n_xyzw[3])*T[:,0])
    v = rh1 * mu / (n_xyzw[1] + n_xyzw[3])
    return pk, v


def mod_rt_iter(T, n_xyzw, p_old, v_old, mu0=0):
    """does a single iteration of the modified RT algorithm
    """
    rk, mu = _stepB(T, n_xyzw, p_old, v_old, mu0)
    rh1 = _r_h1(T, p_old, v_old)
    p_new, v_new = _stepC(T, rk, mu, n_xyzw, rh1)
    return p_new, v_new, mu


def modified_rt(T, p_old, v_old, n_iter, mu0=0, verbose=False):
    """the modified RT algorithm
    the input array, T, should have 5 columns like [t, x, y, z, w]
    (as in the 'Data' side of Table 1 in Soon & Woodrofe)

    p_old, v_old should be initial guesses for (p, v), resp.
    mu0 is an initial guess for the intermediate, mu, parameter
    """
    delp = numpy.zeros((n_iter))
    delv = delp.copy()
    n_xyzw = numpy.sum(T, axis=0)[1:]
    if verbose:
        iterator = tqdm(range(n_iter), total=n_iter)
    else:
        iterator = range(n_iter)
    for i in iterator:
        # do the iteration
        p_new, v_new, mu_new = mod_rt_iter(T, n_xyzw, p_old, v_old, mu0)
        # compute change in parameters
        dp = numpy.sqrt(numpy.sum(numpy.square(p_new-p_old)))
        dv = numpy.abs(v_new - v_old)
        delp[i], delv[i] = dp, dv
        # new is next iter's old
        p_old, v_old, mu0 = p_new.copy(), v_new.copy(), mu_new
    return (p_new, v_new), (delp, delv)


def initialize_pv(T):
    """make initial guess for p, v parameters based on suggestion from paper
    """
    n_xyzw = numpy.sum(T, axis=0)[1:]
    p = numpy.ones_like(T[:,0]) / T.shape[0]
    v = n_xyzw[-1] / numpy.sum(n_xyzw)
    return p, v
