import numpy


def lifetime_array_to_rt_input(L):
    """convert input lifetime array to correct format for the RT algorithms
    input should be 3 columns, 1 row per event:
       [lifetime, left_censored, right_censored]
    output will have 5, 1 row per unique lifetime:
       [lifetime, x, y, z, w]
    """
    times = numpy.unique(L[:,0])
    T = numpy.vstack([_make_rt_input_row(L, t) for t in times])
    return T


def _make_rt_input_row(L, t):
    l = L[L[:,0]==t,1:]
    x, y, z, w = 0, 0, 0, 0
    for ri in range(l.shape[0]):
        if l[ri,0]:
            if l[ri,1]:  # left & right censored
                w += 1
            else:  # left censored only
                y += 1
        else:
            if l[ri,1]:  # right censored only
                z += 1
            else:  # no censoring
                x += 1
    return [t, x, y, z, w]
