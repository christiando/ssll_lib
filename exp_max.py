"""
Functions for performing the expectation maximisation algorithm over the
observed spike-pattern rates and the natural parameters. These functions
use 'call by reference' in that, rather than returning a result, they update
the data referred to by their parameters.

---

This code implements approximate inference methods for State-Space Analysis of
Spike Correlations (Shimazaki et al. PLoS Comp Bio 2012). It is an extension of
the existing code from repository <https://github.com/tomxsharp/ssll> (For
Matlab Code refer to <http://github.com/shimazaki/dynamic_corr>). We
acknowledge Thomas Sharp for providing the code for exact inference.

In this library are additional methods provided to perform the State-Space
Analysis approximately. This includes pseudolikelihood, TAP, and Bethe
approximations. For details see: <http://arxiv.org/abs/1607.08840>

Copyright (C) 2016

Authors of the extensions: Christian Donner (christian.donner@bccn-berlin.de)
                           Hideaki Shimazaki (shimazaki@brain.riken.jp)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy
import pdb

import probability
import transforms
import max_posterior


CONVERGED = 1e-4



def e_step(emd):
    """
    Computes the posterior (approximated as a multivariate Gaussian
    distribution) of the natural parameters of observed spike patterns, given
    the state-transition hyperparameters. Firstly performs a `forward'
    iteration, in which the filter posterior density at time t is determined
    from the observed patterns at time t and the one-step prediction density at
    time t-1. Secondly performs a `backward' iteration, in which these
    sequential filter estimates are smoothed over time.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Compute the 'forward' filter density
    e_step_filter(emd)
    # Compute the 'backward' smooth density
    e_step_smooth(emd)


def e_step_filter(emd):
    """
    Computes the one-step-prediction density and the filter density in the
    expectation step.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Iterate forwards over each timestep, computing filter density
    emd.theta_f[0,:], emd.sigma_f[0,:] = max_posterior.run(emd, 0)
    for i in range(1, emd.T):
        # Compute one-step prediction density
        emd.theta_o[i,:] = numpy.dot(emd.F, emd.theta_f[i-1,:])
        # Computation for exact case with full covariance matrix
        if emd.param_est_eta == 'exact':
            tmp = numpy.dot(emd.F, emd.sigma_f[i-1,:,:])
            emd.sigma_o[i,:,:] = numpy.dot(tmp, emd.F.T) + emd.Q
            # Compute inverse of one-step prediction covariance
            emd.sigma_o_inv[i,:,:] = numpy.linalg.inv(emd.sigma_o[i,:,:])
        # Computation for approximate case with diagonal covariance matrix
        else:
            emd.sigma_o[i,:] = emd.sigma_f[i-1,:] + emd.Q.diagonal()
            emd.sigma_o_inv[i] = 1./emd.sigma_o[i]
        # Get MAP estimate of filter density
        emd.theta_f[i,:], emd.sigma_f[i,:] = max_posterior.run(emd, i)


def e_step_smooth(emd):
    """
    Computes smooth density in the expectation step.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Initialise the smoothed theta and sigma values
    emd.theta_s[-1] = emd.theta_f[-1]
    emd.sigma_s[-1] = emd.sigma_f[-1]
    if emd.param_est_eta == 'exact':
        # Iterate backwards over each timestep, computing smooth density
        for i in reversed(range(emd.T - 1)):
            # Compute the A matrix
            a = numpy.dot(emd.sigma_f[i], emd.F.T)
            A = numpy.dot(a, emd.sigma_o_inv[i+1])
            # Compute the backward-smoothed means
            tmp = numpy.dot(A, emd.theta_s[i+1,:] - emd.theta_o[i+1,:])
            emd.theta_s[i,:] = emd.theta_f[i,:] + tmp
            # Compute the backward-smoothed covariances
            tmp = numpy.dot(A, emd.sigma_s[i+1] - emd.sigma_o[i+1])
            tmp = numpy.dot(tmp, A.T)
            emd.sigma_s[i] = emd.sigma_f[i] + tmp
            # Compute the backward-smoothed lag-one covariances
            emd.sigma_s_lag[i+1] = numpy.dot(A, emd.sigma_s[i+1,:])
    else:
        for i in reversed(range(emd.T - 1)):
            # Compute the A matrix
            a = numpy.dot(numpy.diag(emd.sigma_f[i]), emd.F.T)
            A = numpy.dot(a, numpy.diag(emd.sigma_o_inv[i+1]))
            # Compute the backward-smoothed means
            tmp = numpy.dot(A, emd.theta_s[i+1,:] - emd.theta_o[i+1,:])
            emd.theta_s[i,:] = emd.theta_f[i,:] + tmp
            # Compute the backward-smoothed covariances
            tmp = numpy.dot(A, numpy.diag(emd.sigma_s[i+1] - emd.sigma_o[i+1]))
            tmp = numpy.dot(tmp, A.T)
            emd.sigma_s[i] = numpy.diagonal(numpy.diag(emd.sigma_f[i]) + tmp)
            # Compute the backward-smoothed lag-one covariances
            emd.sigma_s_lag[i+1] = numpy.dot(A, numpy.diag(emd.sigma_s[i+1])).diagonal()


def m_step(emd, stationary='None'):
    """
    Computes the optimised hyperparameters of the natural parameters of the
    posterior distributions over time. `Q' is the covariance matrix of the
    transition probability distribution. `F' is the autoregressive parameter of
    the state transitions, but it is kept constant in this implementation.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param stationary:
        If 'all' stationary on all thetas is assumed.
    """
    # Update the initial mean of the one-step-prediction density
    emd.theta_o[0, :] = emd.theta_s[0, :]
    # Compute the state-transition hyperparameter
    m_step_Q(emd, stationary)
    #m_step_F(emd)


def m_step_F(emd):
    """
    Computes the optimised autogregressive hyperparameter `F' of the natural
    parameters of the posterior distributions over time. See equation 39 of
    the source paper for details.

    NB: This function is not called in this implementation because the
    autoregressive parameter is kept constant.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Set up temporary-results arrays
    a = numpy.zeros((emd.D, emd.D))
    b = numpy.zeros((emd.D, emd.D))
    # Sum partial results over each timestep
    for i in range(1, emd.T):
        a += emd.sigma_s_lag[i,:,:] +\
             numpy.outer(emd.theta_s[i,:], emd.theta_s[i-1,:])
        b += emd.sigma_s[i-1,:,:] +\
             numpy.outer(emd.theta_s[i-1,:], emd.theta_s[i-1,:])
    # Dot the results
    emd.F = numpy.dot(a, numpy.linalg.inv(b))


def m_step_Q(emd, stationary):
    """
    Computes the optimised state-transition covariance hyperparameters `Q' of
    the natural parameters of the posterior distributions over time. Here
    just one single scalar is considered

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param stationary:
        If 'all' stationary on all thetas is assumed.
    """
    inv_lmbda = 0
    if emd.param_est_eta == 'exact':
        for i in range(1, emd.T):
            lag_one_covariance = emd.sigma_s_lag[i, :, :]
            tmp = emd.theta_s[i, :] - emd.theta_s[i - 1, :]
            inv_lmbda += numpy.trace(emd.sigma_s[i, :, :]) - \
                         2 * numpy.trace(lag_one_covariance) + \
                         numpy.trace(emd.sigma_s[i - 1, :, :]) + \
                         numpy.dot(tmp, tmp)
        emd.Q = inv_lmbda / emd.D / (emd.T - 1) * numpy.identity(emd.D)
    else:
        for i in range(1, emd.T):
            lag_one_covariance = emd.sigma_s_lag[i, :]
            tmp = emd.theta_s[i, :] - emd.theta_s[i - 1, :]
            inv_lmbda += numpy.sum(emd.sigma_s[i]) - \
                         2 * numpy.sum(lag_one_covariance) + \
                         numpy.sum(emd.sigma_s[i - 1]) + \
                         numpy.dot(tmp, tmp)
        emd.Q = inv_lmbda / emd.D / (emd.T - 1) * \
                numpy.identity(emd.D)
    if stationary == 'all':
        emd.Q = numpy.zeros(emd.Q.shape)


def m_step_Q2(emd, stationary):
    """
    Computes the optimised state-transition covariance hyperparameters `Q' of
    the natural parameters of the posterior distributions over time. Two
    different scalars for theta_1 and theta_2 are considered

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param stationary:
        If 'all' stationary on all thetas is assumed.
    """
    inv_lmbda1 = 0.
    inv_lmbda2 = 0.
    # Computation for exact case with full covariance matrix
    if emd.param_est_eta == 'exact':
        for i in range(1, emd.T):
            # Loading saved lag-one smoother
            lag_one_covariance = emd.sigma_s_lag[i,:,:]
            tmp = emd.theta_s[i,:] - emd.theta_s[i-1,:]
            inv_lmbda1 += numpy.trace(emd.sigma_s[i,:emd.N,:emd.N]) -\
                     2 * numpy.trace(lag_one_covariance[:emd.N,:emd.N])  +\
                     numpy.trace(emd.sigma_s[i-1,:emd.N,:emd.N])  +\
                     numpy.dot(tmp[:emd.N], tmp[:emd.N])
            inv_lmbda2 += numpy.trace(emd.sigma_s[i,emd.N:,emd.N:]) -\
                     2 * numpy.trace(lag_one_covariance[emd.N:,emd.N:])  +\
                     numpy.trace(emd.sigma_s[i-1,emd.N:,emd.N:])  +\
                     numpy.dot(tmp[emd.N:], tmp[emd.N:])

        emd.Q[:emd.N,:emd.N] = inv_lmbda1 / emd.N / (emd.T - 1) * numpy.identity(emd.N)
        if emd.order > 1:
            emd.Q[emd.N:,emd.N:] = inv_lmbda2 / (emd.D - emd.N) / (emd.T - 1) * numpy.identity(emd.D - emd.N)

        if stationary == 'all':
            emd.Q[:, :] = 0
    # Computation for approximate case with diagonal covariance matrix
    else:
        for i in range(1, emd.T):
            # Loading saved lag-one smoother
            lag_one_covariance = emd.sigma_s_lag[i, :]
            tmp = emd.theta_s[i, :] - emd.theta_s[i - 1, :]
            inv_lmbda1 += numpy.sum(emd.sigma_s[i, :emd.N]) - \
                          2 * numpy.sum(lag_one_covariance[:emd.N]) + \
                          numpy.sum(emd.sigma_s[i - 1, :emd.N]) + \
                          numpy.inner(tmp[:emd.N], tmp[:emd.N])
            inv_lmbda2 += numpy.sum(emd.sigma_s[i, emd.N:]) - \
                          2 * numpy.sum(lag_one_covariance[emd.N:]) + \
                          numpy.sum(emd.sigma_s[i - 1, emd.N:]) + \
                          numpy.inner(tmp[emd.N:], tmp[emd.N:])

        emd.Q[:emd.N, :emd.N] = inv_lmbda1 / emd.N / (emd.T - 1) * \
                                numpy.identity(emd.N)
        if emd.order > 1:
            emd.Q[emd.N:, emd.N:] = inv_lmbda2 / (emd.D - emd.N) / \
                                    (emd.T - 1) * numpy.identity(emd.D - emd.N)
        if stationary == 'all':
            emd.Q[:] = 0

def m_step_Q3(emd, stationary):
    """
    Computes the optimised state-transition covariance hyperparameters `Q' of
    the natural parameters of the posterior distributions over time. For each
    individual theta a hyperparameter is considered.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param stationary:
        If 'all' stationary on all thetas is assumed.
    """
    lmbda = numpy.zeros([emd.Q.shape[0]])
    # Computation for exact case with full covariance matrix
    if emd.param_est_eta == 'exact':
        for i in range(1, emd.T):
            # Loading saved lag-one smoother
            lag_one_covariance = emd.sigma_s_lag[i,:,:]
            tmp = emd.theta_s[i,:] - emd.theta_s[i-1,:]
            lmbda += numpy.diagonal(emd.sigma_s[i,:,:]) -\
                     2 * numpy.diagonal(lag_one_covariance[:,:]) +\
                     numpy.diagonal(emd.sigma_s[i-1,:,:]) +\
                     tmp**2

        if stationary == 'all':
            lmbda[:,:] = 0
    # Computation for approximate case with diagonal covariance matrix
    else:
        for i in range(1, emd.T):
            # Loading saved lag-one smoother
            lag_one_covariance = emd.sigma_s_lag[i]
            tmp = emd.theta_s[i,:] - emd.theta_s[i-1,:]
            lmbda += emd.sigma_s[i] -\
                     2 * lag_one_covariance  +\
                     emd.sigma_s[i-1]  +\
                     tmp**2

        if stationary == 'all':
            lmbda[:] = 0

    emd.Q = numpy.diag(lmbda / (emd.T - 1))
