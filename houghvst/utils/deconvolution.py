# -*- coding: utf-8 -*-
"""
Extract neural activity from a fluorescence trace using a constrained
deconvolution approach

Code obtained from:
https://github.com/simonsfoundation/CaImAn

Created on Tue Sep  1 16:11:25 2015
@author: Eftychios A. Pnevmatikakis, based on an implementation by
T. Machado, Andrea Giovannucci & Ben Deverett
"""

import cvxpy as cvx
import numpy as np
import scipy.signal
import scipy.sparse


def deconvolve_black_box(fluor):
    g, sn = estimate_parameters(fluor, p=2, fudge_factor=.98)
    res = deconvolve(fluor, g, sn)
    fluor_denoised = res[0]
    spikes = res[5]
    return fluor_denoised, spikes


def deconvolve(fluor, g, sn, b=None, c1=None, bas_nonneg=True, solvers=None):
    """
    Solves the deconvolution problem using the cvxpy package and the
    ECOS/SCS library.

    Parameters:
    -----------
    fluor: ndarray
        fluorescence trace

    g: list of doubles
        parameters of the autoregressive model, cardinality equivalent
        to p

    sn: double
        estimated noise level

    b: double
        baseline level. If None it is estimated.

    c1: double
        initial value of calcium. If None it is estimated.

    bas_nonneg: boolean
        should the baseline be estimated

    solvers: tuple of two strings
        primary and secondary solvers to be used. Can be choosen
        between ECOS, SCS, CVXOPT

    Returns:
    --------

    c: estimated calcium trace

    b: estimated baseline

    c1: esimtated initial calcium value

    g: esitmated parameters of the autoregressive model

    sn: estimated noise level

    sp: estimated spikes

    Raise:
    -----
    ImportError('cvxpy solver requires installation of cvxpy. Not
    working in windows at the moment.')

    ValueError('Problem solved suboptimally or unfeasible')

    """
    # todo: check the result and gen_vector vars
    if solvers is None:
        solvers = ['ECOS', 'SCS']

    T = fluor.size

    # construct deconvolution matrix  (sp = G*c)
    G = scipy.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))

    for i, gi in enumerate(g):
        G += scipy.sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))

    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gd_vec = np.max(gr) ** np.arange(T)  # decay vector for initial fluorescence

    c = cvx.Variable(T)  # calcium at each time step
    constraints = []
    cnt = 0
    if b is None:
        flag_b = True
        cnt += 1
        b = cvx.Variable(1)  # baseline value
        if bas_nonneg:
            b_lb = 0
        else:
            b_lb = np.min(fluor)
        constraints.append(b >= b_lb)
    else:
        flag_b = False

    if c1 is None:
        flag_c1 = True
        cnt += 1
        c1 = cvx.Variable(1)  # baseline value
        constraints.append(c1 >= 0)
    else:
        flag_c1 = False

    constraints.append(G * c >= 0)

    behavior = cvx.norm(-c + fluor - b - gd_vec * c1, 2)

    try:
        # minimize number of spikes (using the l1-norm as a proxy)
        objective = cvx.Minimize(cvx.sum_entries(G * c))

        thres_noise = sn * np.sqrt(fluor.size)
        prob = cvx.Problem(objective, constraints + [behavior <= thres_noise])
        prob.solve(solver=solvers[0])

        if prob.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError('Problem solved suboptimally or unfeasible')

        print(('PROBLEM STATUS:' + prob.status))

    except (ValueError, cvx.SolverError):  # solvers fail to solve the problem

        lam = sn / 500
        objective = cvx.Minimize(behavior + lam * cvx.sum_entries(G * c))
        prob = cvx.Problem(objective, constraints)

        try:
            # print('TRYING AGAIN', solvers[0])
            prob.solve(solver=solvers[0])
        except:
            print((solvers[0] + ' DID NOT WORK TRYING ' + solvers[1]))
            prob.solve(solver=solvers[1])

        if prob.status not in ['optimal', 'optimal_inaccurate']:
            print(('PROBLEM STATUS:' + prob.status))
            sp = fluor
            c = fluor
            b = 0
            c1 = 0
            return c, b, c1, g, sn, sp

    sp = np.squeeze(np.asarray(G * c.value))
    c = np.squeeze(np.asarray(c.value))
    if flag_b:
        b = np.squeeze(b.value)
    if flag_c1:
        c1 = np.squeeze(c1.value)

    return c, b, c1, g, sn, sp


def estimate_parameters(fluor, p=2, sn=None, g=None, range_ff=[0.25, 0.5],
                        method='logmexp', lags=5, fudge_factor=1.):
    """
    Estimate noise standard deviation and AR coefficients if they are
    not present

    Parameters:
    -----------

    p: positive integer
        order of AR system

    sn: float
        noise standard deviation, estimated if not provided.

    lags: positive integer
        number of additional lags where he autocovariance is computed

    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is
        averaged

    method: string
        method of averaging: Mean, median, exponentiated mean of
        logvalues (default)

    fudge_factor: float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
    """

    if sn is None:
        sn = compute_noise_power(fluor, range_ff, method)

    if g is None:
        if p == 0:
            g = np.array(0)
        else:
            g = estimate_time_constant(fluor, p, sn, lags, fudge_factor)

    return g, sn


def estimate_time_constant(fluor, p=2, sn=None, lags=5, fudge_factor=1.):
    """
    Estimate AR model parameters through the autocovariance function

    Inputs:
    --------

    fluor        : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.

    p            : positive integer
        order of AR system

    sn           : float
        noise standard deviation, estimated if not provided.

    lags         : positive integer
        number of additional lags where he autocovariance is computed

    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias

    Returns:
    -----------

    g       : estimated coefficients of the AR process
    """

    if sn is None:
        sn = compute_noise_power(fluor)

    lags += p
    xc = axcov(fluor, lags)
    xc = xc[:, np.newaxis]

    A = scipy.linalg.toeplitz(xc[lags + np.arange(lags)],
                              xc[lags + np.arange(p)]) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[lags + 1:])[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = (gr + gr.conjugate()) / 2.
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def compute_noise_power(fluor, range_ff=[0.25, 0.5], method='logmexp'):
    """    
    Estimate noise power through the power spectral density over the
    range of large frequencies    

    Inputs:
    ----------

    fluor    : nparray
        One dimensional array containing the fluorescence intensities
        with one entry per time-bin.

    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is
        averaged  

    method   : string
        method of averaging: Mean, median, exponentiated mean of
        logvalues (default)

    Returns:
    -----------
    sn       : noise standard deviation
    """

    ff, Pxx = scipy.signal.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda x: np.sqrt(np.mean(x / 2.)),
        'median': lambda x: np.sqrt(np.median(x / 2.)),
        'logmexp': lambda x: np.sqrt(np.exp(np.mean(np.log(x / 2.))))
    }[method](Pxx_ind)

    return sn


def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag

    Parameters:
    ----------
    data : array
        Array containing fluorescence data

    maxlag : int
        Number of lags to use in autocovariance calculation

    Returns:
    -------
    axcov : array
        Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(xcov / T)


def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than
    abs(value).

    Parameters:
    ----------
    value : int

    Returns:
    -------
    exponent : int
    """
    return int(np.ceil(np.log2(np.abs(value))))
