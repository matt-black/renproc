import numpy
import scipy


def fit_distribution(t, p, dist, curve='pdf', p0=None):
    """fit input data of times (t) and corresp. probability (p) to a distribution
    by fitting either the PDF (`curve='pdf'`) or CDF (`curve='cdf'`)
    """
    assert dist in __DISTRIBUTIONS
    assert curve in ('pdf', 'cdf')
    # calculate mean/variance so we have good starting guess
    mu = numpy.sum(t * p)
    var = numpy.sum(p * numpy.square(t - mu))
    if curve == 'cdf':
        # make sure its sorted
        ind = t.argsort()
        t, p = t[ind], p[ind]
        y = numpy.cumsum(p)
    else:
        y = p
    if dist == 'exponential':
        d = Exponential(None)
        p0 = [1/mu] if p0 is None else p0
    elif dist == 'delayexp':
        d = DelayedExponential(None, None)
        if p0 is None:
            p0 = [1/mu, numpy.amin(t[p>0.0001])]
    elif dist == 'weibull':
        d = Weibull(None, None)
        p0 = [mu, 1] if p0 is None else p0
    elif dist == 'gamma':
        d = Gamma(None, None)
        p0 = [1/mu, 1] if p0 is None else p0
    elif dist == 'pareto':
        d = Pareto(None, None)
        p0 = [numpy.amin(t[p>0.0001]), 0.5]
    elif dist == 'burr':
        d = Burr(None, None, None)
        p0 = [0.75,]*3
    else:
        raise ValueError('invalid distribution')
    fun = d.cdf if curve == 'cdf' else d.pdf
    return scipy.optimize.curve_fit(fun, t, y, p0=p0)


__DISTRIBUTIONS = ('exponential', 'delayexp', 'weibull', 'gamma',
                   'pareto', 'burr')


class Distribution(object):
    def __init__(self):
        pass

    def pdf(self):
        raise NotImplementedError()

    def cdf(self):
        raise NotImplementedError()

    def mean(self):
        raise NotImplementedError()
    
    def median(self):
        raise NotImplementedError()

    def variance(self):
        raise NotImplementedError()
    

class Exponential(Distribution):
    def __init__(self, _lambda):
        self._lambda = _lambda

    def pdf(self, t, _lambda=None):
        _lambda = self._lambda if _lambda is None else _lambda
        return _lambda * numpy.exp(-_lambda * t)

    def cdf(self, t, _lambda=None):
        _lambda = self._lambda if _lambda is None else _lambda
        return 1 - numpy.exp(-_lambda * t)

    def mean(self, _lambda=None):
        _lambda = self._lambda if _lambda is None else _lambda
        return 1/_lambda

    def median(self, _lambda=None):
        _lambda = self._lambda if _lambda is None else _lambda
        return numpy.log(2)/_lambda

    def variance(self, _lambda=None):
        _lambda = self._lambda if _lambda is None else _lambda
        return 1 / numpy.square(_lambda)
    

class DelayedExponential(Distribution):
    def __init__(self, _lambda, _delay):
        self._lambda = _lambda
        self._delay = _delay

    def pdf(self, t, _lambda=None, _delay=None):
        _lambda = self._lambda if _lambda is None else _lambda
        _delay = self._delay if _delay is None else _delay
        filt = t >= _delay
        zero = numpy.zeros((numpy.sum(numpy.logical_not(filt)),))
        rest = Exponential(_lambda).pdf(t[filt]-_delay)
        return numpy.concatenate([zero, rest])

    def cdf(self, t, _lambda=None, _delay=None):
        _lambda = self._lambda if _lambda is None else _lambda
        _delay = self._delay if _delay is None else _delay
        filt = t >= _delay
        zero = numpy.zeros((numpy.sum(numpy.logical_not(filt)),))
        rest = Exponential(_lambda).cdf(t[filt]-_delay)
        return numpy.concatenate([zero, rest])

    def mean(self, _lambda=None, _delay=None):
        _lambda = self._lambda if _lambda is None else _lambda
        _delay = self._delay if _delay is None else _delay
        return Exponential(_lambda).mean() + _delay

    def median(self, _lambda=None, _delay=None):
        raise NotImplementedError()

    def variance(self, _lambda=None, _delay=None):
        raise NotImplementedError()
    
    
class Weibull(Distribution):
    def __init__(self, _lambda, _k):
        self._lambda = _lambda
        self._k = _k

    def pdf(self, t, _lambda=None, _k=None):
        _lambda = self._lambda if _lambda is None else _lambda
        _k = self._k if _k is None else _k
        return (_k/_lambda)*numpy.power(t/_lambda, _k-1) * \
            numpy.exp(-numpy.power(t/_lambda, k))

    def cdf(self, t, _lambda=None, _k=None):
        _lambda = self._lambda if _lambda is None else _lambda
        _k = self._k if _k is None else _k
        return 1 - numpy.exp(-numpy.power(t/_lambda, _k))

    def mean(self, _lambda=None, _k=None):
        _lambda = self._lambda if _lambda is None else _lambda
        _k = self._k if _k is None else _k
        return _lambda * scipy.special.gamma(1 + 1/_k)

    def median(self, _lambda=None, _k=None):
        _lambda = self._lambda if _lambda is None else _lambda
        _k = self._k if _k is None else _k
        return _lambda * numpy.power(numpy.log(2), 1/k)

    def variance(self, _lambda=None, _k=None):
        _lambda = self._lambda if _lambda is None else _lambda
        _k = self._k if _k is None else _k
        gam2 = scipy.special.gamma(1 + 2/_k)
        gam1 = scipy.special.gamma(1 + 1/_k)
        return numpy.square(_lambda) * (gam2 - numpy.square(gam1))
    

class Gamma(Distribution):
    def __init__(self, _alpha, _beta):
        self._alpha = _alpha
        self._beta = _beta

    def pdf(self, t, _alpha=None, _beta=None):
        _alpha = self._alpha if _alpha is None else _alpha
        _beta = self._beta if _beta is None else _beta
        return numpy.power(_beta, _alpha)/scipy.special.gamma(alpha) * \
            numpy.power(t, _alpha-1) * numpy.exp(-_beta*t)

    def cdf(self, t, _alpha=None, _beta=None):
        _alpha = self._alpha if _alpha is None else _alpha
        _beta = self._beta if _beta is None else _beta
        return scipy.special.gammainc(_alpha, _beta*t) / \
            scipy.special.gamma(_alpha)

    def mean(self, _alpha=None, _beta=None):
        _alpha = self._alpha if _alpha is None else _alpha
        _beta = self._beta if _beta is None else _beta
        return _alpha / _beta

    def median(self, _alpha=None, _beta=None):
        raise ValueError('no closed form')

    def variance(self, _alpha=None, _beta=None):
        _alpha = self._alpha if _alpha is None else _alpha
        _beta = self._beta if _beta is None else _beta
        return _alpha / numpy.square(_beta)


class Pareto(Distribution):
    def __init__(self, _x_m, _alpha):
        self._x_m = _x_m
        self._alpha = _alpha

    def pdf(self, t, _x_m=None, _alpha=None):
        _x_m = self._x_m if _x_m is None else _x_m
        _alpha = self._alpha if _alpha is None else _alpha
        return (_alpha * numpy.power(_x_m, _alpha)) / numpy.power(t, _alpha+1)

    def cdf(self, t, _x_m=None, _alpha=None):
        _x_m = self._x_m if _x_m is None else _x_m
        _alpha = self._alpha if _alpha is None else _alpha
        filt = t >= _x_m
        too_low = numpy.zeros((numpy.sum(numpy.logical_not(filt)),))
        rest = 1 - numpy.power(_x_m/t[filt], _alpha)
        return numpy.concatenate([too_low, rest])

    def mean(self, _x_m=None, _alpha=None):
        _x_m = self._x_m if _x_m is None else _x_m
        _alpha = self._alpha if _alpha is None else _alpha
        if _alpha > 1:
            return (_alpha * _x_m) / (_alpha - 1)
        else:
            return numpy.inf

    def median(self, _x_m=None, _alpha=None):
        _x_m = self._x_m if _x_m is None else _x_m
        _alpha = self._alpha if _alpha is None else _alpha
        return _x_m * numpy.power(2, 1/_alpha)

    def variance(self, _x_m=None, _alpha=None):
        _x_m = self._x_m if _x_m is None else _x_m
        _alpha = self._alpha if _alpha is None else _alpha
        if _alpha > 2:
            return (numpy.square(_x_m)*_alpha) / \
                (numpy.square(_alpha-1)*(_alpha-2))
        else:
            return numpy.inf
        

class Burr(Distribution):
    def __init__(self, _alpha, _c, _k):
        self._alpha = _alpha
        self._c = _c
        self._k = _k

    def pdf(self, t, _alpha=None, _c=None, _k=None):
        _alpha = self._alpha if _alpha is None else _alpha
        _c = self._c if _c is None else _c
        _k = self._k if _k is None else _k
        return (_k * _c * numpy.power(t/_alpha, _c-1)) / \
            (_alpha*numpy.power(1 + numpy.power(t/_alpha, _c), _k+1))

    def cdf(self, t, _alpha=None, _c=None, _k=None):
        _alpha = self._alpha if _alpha is None else _alpha
        _c = self._c if _c is None else _c
        _k = self._k if _k is None else _k
        return 1 - numpy.power(1 + numpy.power(t, _c), -_k)

    def mean(self, _alpha=None, _c=None, _k=None):
        _alpha = self._alpha if _alpha is None else _alpha
        _c = self._c if _c is None else _c
        _k = self._k if _k is None else _k
        return -k * scipy.special.beta(_k-1/_c, 1+1/_c)

    def median(self, _alpha=None, _c=None, _k=None):
        _alpha = self._alpha if _alpha is None else _alpha
        _c = self._c if _c is None else _c
        _k = self._k if _k is None else _k
        return numpy.power(numpy.power(2, 1/_k) - 1, 1/_c)
    
