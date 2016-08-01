"""
This file contains all methods that are concerned with the Bethe approximation.

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


def construct_fisher_diag(eta, N):
    eta1 = eta[:N]
    eta2 = eta[N:]
    ddllk_diag = numpy.empty(eta.shape)
    ddllk_diag[:N] = eta1 - eta1**2
    eta1_outer = numpy.outer(eta1, eta1)
    triu_idx = numpy.triu_indices(N, 1)
    ddllk_diag[N:] = eta2 - eta1_outer[triu_idx]
    ddllk_diag = eta - eta**2
    return ddllk_diag


def compute_eta_CCCP(theta, N, return_psi=False):
    """ CCCP Algorithm to find solution for Bethe free energy
    [Yuille, 2002 Neural Comp.]

    :param numpy.ndarray theta:
        (d,) dimensional array with natural parameters in it
    :param int N:
        Number of cells
    :param bool return_psi:
        If true the appoximation log partition function is returned in addition
        (Default=False)
    :returns:
        (d,) dimensional array with approximated etas
    """
    triu_idx = numpy.triu_indices(N, 1)
    diag_idx = numpy.diag_indices(N)
    # Transform thetas
    theta1 = theta[:N]
    theta2 = numpy.zeros([N, N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    # Get unnormalized probs
    psi_i = numpy.ones([N, 2])
    psi_i[:,1] = numpy.exp(theta1)
    phi_ij = numpy.ones([N, N, 4])
    phi_ij[:, :, 1] = numpy.exp(theta1[:, numpy.newaxis])
    phi_ij[:, :, 2] = numpy.exp(theta1[:, numpy.newaxis].T)
    phi_ij[:, :, 3] = numpy.exp(theta1[:, numpy.newaxis] +
                                theta1[:, numpy.newaxis].T + theta2)
    phi_ij[diag_idx[0], diag_idx[1], :] = 1
    # Initialize beliefs and Lagrange multipliers
    b_i = .5*numpy.ones([N, 2])
    b_ij = .25*numpy.ones([N, N, 4])
    b_ij[diag_idx[0], diag_idx[1], :] = 0
    lambda_ij = numpy.zeros([N, N, 2])
    gamma_ij = numpy.zeros([N, N])
    # Start CCCP
    eta1, eta2, bethe_energy = outer_loop(b_i, b_ij, phi_ij, psi_i, lambda_ij,
                                          gamma_ij, N)
    # Reshape the expectation parameters and return
    eta = numpy.empty(theta.shape)
    eta[:N] = eta1[:, 1]
    eta[N:] = eta2[triu_idx[0], triu_idx[1], 3]
    # Instead of the Bethe energy one could just use psi=-log(p(0))
    # bethe_free = -(numpy.sum(numpy.log(b_ij[triu_idx[0], triu_idx[1], 0])) -
    #                numpy.sum(((N - 1) - 1) * numpy.log(b_i[:, 0])))
    if return_psi:
        return eta, -bethe_energy
    else:
        return eta


def outer_loop(b_i, b_ij, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Outer loop of CCCP to update beliefs

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in
        [:,1]
    :param numpy.ndarray b_ij:
        [c,c,4] array with second order beliefs for pair patterns
        (0,0),(1,0),(0,1) and (1,1)
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for
        pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with Lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with Lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        first order, second order believes, bethe energy

    """
    # Compute Bethe energy
    bethe_E = bethe_free_energy(b_i, b_ij, psi_i, phi_ij, N)
    conv_crit = numpy.inf
    bethe = [bethe_E]
    triu_idx = numpy.triu_indices(N, 1)
    while conv_crit > 1e-5:
        # Until convergence update Lagrange multipliers and beliefs
        lambda_ij, gamma_ij = inner_loop(b_i, b_ij, phi_ij, psi_i, lambda_ij,
                                         gamma_ij, N)
        b_i, b_ij = update_beliefs(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N)
        # Compute Bethe energy
        bethe_E_old = bethe_E
        # Compute bethe free energy
        bethe_E = bethe_free_energy(b_i, b_ij, psi_i, phi_ij, N)
        bethe.append(bethe_E)
        conv_crit = numpy.absolute((bethe_E_old - bethe_E) / bethe_E_old)

    return b_i, b_ij, bethe_E


def inner_loop(b_i, b_ij, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Inner loop of CCCP to find lagrange multipliers

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in
        [:,1]
    :param numpy.ndarray b_ij:
        [c,c,4] array with second order beliefs for pair patterns
        (0,0),(1,0),(0,1) and (1,1)
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for
        pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """
    # Compute energy
    dual_E = compute_dual_energy(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N)
    conv_crit = numpy.inf

    while conv_crit > 1e-10:
        # Until not converged update lagrange multipliers
        lambda_ij = update_lambda(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N)
        gamma_ij = update_gamma(phi_ij, lambda_ij, N)
        # Compute dual energy
        dual_E_old = dual_E
        dual_E = compute_dual_energy(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N)
        conv_crit = (dual_E_old - dual_E) / dual_E_old

    # Return
    return lambda_ij, gamma_ij


def update_lambda(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Update lambdas.

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in
        [:,1]
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for
        pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """

    # Diagonal indices
    diag_idx = numpy.diag_indices(N)
    p = diag_idx[0]
    # Iterate through each (off-)diagonal

    for i in range(1, N):
        q = (diag_idx[1] + i) % N
        # Update lambda_ij(0)
        exp2lambda0 = (numpy.sum(phi_ij[p, q, :2] * \
                                 numpy.exp(-lambda_ij[q, p] - gamma_ij[p, q][:,numpy.newaxis]), axis=1)) \
                                 /(psi_i[q, 0] * (b_i[q, 0] / psi_i[q, 0]) ** (N - 1) * numpy.exp((N - 1) + \
                                 numpy.sum(lambda_ij[:, q, 0], axis=0) - lambda_ij[p, q, 0]))
        lambda_ij[p, q, 0] = .5 * numpy.log(exp2lambda0)
        # Update lambda_ij(1)
        exp2lambda1 = (numpy.sum(phi_ij[p, q, 2:] * \
                                 numpy.exp(-lambda_ij[q, p] - gamma_ij[p, q][:,numpy.newaxis]), axis=1)) \
                                 /(psi_i[q, 1] * (b_i[q, 1] / psi_i[q, 1]) ** (N - 1) * numpy.exp((N - 1) + \
                                 numpy.sum(lambda_ij[:, q, 1], axis=0) - lambda_ij[p, q, 1]))
        lambda_ij[p, q, 1] = .5 * numpy.log(exp2lambda1)

    # Return lambdas
    return lambda_ij


def update_gamma(phi_ij, lambda_ij, N):
    """ Update gammas.

    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for
        pair patterns
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """

    diag_idx = numpy.diag_indices(N)
    # Compute lambdas for all pair states (0,0),(1,0),(0,1) and (1,1)
    lmbd_tmp = numpy.zeros([N, N, 4])
    lmbd_tmp[:, :, 0] = lambda_ij[:, :, 0] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 1] = lambda_ij[:, :, 0] + lambda_ij[:, :, 1].T
    lmbd_tmp[:, :, 2] = lambda_ij[:, :, 1] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 3] = lambda_ij[:, :, 1] + lambda_ij[:, :, 1].T
    # Compute new gammas
    gamma_new = numpy.log(numpy.sum(phi_ij * numpy.exp(-1. - lmbd_tmp), axis=2))
    gamma_new[diag_idx] = 0
    # Return
    return gamma_new


def update_beliefs(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Update beliefs.

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in
        [:,1]
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for
        pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij
    """
    lmbd_tmp = numpy.zeros([N, N, 4])
    lmbd_tmp[:, :, 0] = lambda_ij[:, :, 0] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 1] = lambda_ij[:, :, 0] + lambda_ij[:, :, 1].T
    lmbd_tmp[:, :, 2] = lambda_ij[:, :, 1] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 3] = lambda_ij[:, :, 1] + lambda_ij[:, :, 1].T
    b_ij_new = phi_ij * numpy.exp(-1. - lmbd_tmp -
                                  gamma_ij[:, :, numpy.newaxis])
    b_i_new = psi_i * numpy.exp(-1. + (N - 1.) + numpy.sum(lambda_ij, axis=0)) \
              * (b_i / psi_i) ** (N - 1)

    return b_i_new, b_ij_new


def compute_dual_energy(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Update dual energy (maximized by inner loop)

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in
        [:,1]

    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for
        pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """
    # Compute lambdas for all pair states (0,0),(1,0),(0,1) and (1,1)
    lmbd_tmp = numpy.zeros([N, N, 4])
    lmbd_tmp[:, :, 0] = lambda_ij[:, :, 0] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 1] = lambda_ij[:, :, 0] + lambda_ij[:, :, 1].T
    lmbd_tmp[:, :, 2] = lambda_ij[:, :, 1] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 3] = lambda_ij[:, :, 1] + lambda_ij[:, :, 1].T
    # Compute dual energy
    dual_energy = -numpy.sum(phi_ij * numpy.exp(-1. - lmbd_tmp -
                                                gamma_ij[:, :, numpy.newaxis]))
    dual_energy -= numpy.sum(
        psi_i * numpy.exp(-1. + (N - 1.) + numpy.sum(lambda_ij, axis=0)) *
        (b_i / psi_i) ** (N - 1))
    dual_energy -= numpy.sum(gamma_ij)
    return dual_energy


def log_marginal_CCCP(emd, period=None):
    """
    Computes the approximate log marginal probability of the observed spike-pattern rates
    by Bethe approximation (CCCP).

    This is just a wrapper function for `log_marginal_raw_CCCP`. It unpacks data
    from the EMD container object and calls that function.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """
    # Unwrap the parameters and call the raw function
    log_p = log_marginal_raw_CCCP(emd.theta_f, emd.theta_o, emd.sigma_f,
                                  emd.sigma_o_inv, emd.y, emd.R, emd.N, period)

    return log_p


def log_marginal_raw_CCCP(theta_f, theta_o, sigma_f, sigma_o_inv, y, R, N,
                          period=None):
    """
    Computes the approximate log marginal probability of the observed spike
    pattern rates by Bethe approximation (CCCP).

    From within SSLL, this function should be accessed by calling
    `log_marginal_CCCP` with the EMD container as a parameter.

    See the container.py for a full description of the parameter properties.

    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """

    if period == None: period = (0, theta_f.shape[0])
    # Initialise
    log_p = 0
    # Iterate over each timestep and compute...
    a, b = 0, 0
    for i in range(period[0], period[1]):
        a += log_likelihood_CCCP(y[i,:], theta_f[i,:], R, N)
        theta_d = theta_f[i,:] - theta_o[i,:]
        b -= numpy.dot(theta_d, sigma_o_inv[i,:]*theta_d)
        b += numpy.sum(numpy.log(sigma_f[i])) +\
             numpy.sum(numpy.log(sigma_o_inv[i]))
    log_p = a + b / 2

    return log_p


def log_likelihood_CCCP(y_t, theta_f_t, R, N):
    """ Computes the log-likelihood with Bethe approximation (CCCP).

    :param numpy.ndarray y_t:
        Frequency of observed patterns for one timestep.
    :param numpy.ndarray theta_f_t:
        Natural parameters of observed patterns for one timestep.
    :param int R:
        Number of trials over which patterns were observed.
    :param int N:
        Number of cells

    :returns:
        Log likelhood of the observed patterns given the natural parameters,
        as a float.
    """
    psi_bethe = compute_eta_CCCP(theta_f_t, N, return_psi=1)[1]
    log_p = R * (numpy.dot(y_t, theta_f_t) - psi_bethe)
    return log_p


def bethe_free_energy(b_i, b_ij, psi_i, phi_ij, N):
    """ Compute Bethe energy. (Minimized by outer loop)

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in
        [:,1]
    :param numpy.ndarray b_ij:
        [c,c,4] array with second order beliefs for pair patterns
        (0,0),(1,0),(0,1) and (1,1)
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij)
        for pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """
    triu_idx = numpy.triu_indices(N, 1)
    # Compute Bethe energy
    bethe_E = numpy.sum(b_ij[triu_idx] * (numpy.log(b_ij[triu_idx]) -
                                          numpy.log(phi_ij[triu_idx]))) \
              - ((N - 1.) - 1.) * numpy.sum(b_i * (numpy.log(b_i) -
                                                   numpy.log(psi_i)))
    return bethe_E


def compute_eta_BP(theta, N, alpha=.5, return_psi=False):
    """ Computes the expectation parameters for given theta according to Bethe
    approximation and belief propagation

    :param numpy.ndarray theta:
        (d,) dimensional array with natural parameters in it
    :param int N:
        Number of cells
    :param float alpha:
        Step size for message update (default=0.5)
    :returns:
        (d,) dimensional array with approximated etas
    """
    # Upper triangle indices
    triu_idx = numpy.triu_indices(N, 1)
    diag_idx = numpy.diag_indices(N)
    # First order theta in square matrix form
    from_idx, to_idx = numpy.meshgrid(numpy.arange(N), numpy.arange(N))
    theta1 = theta[to_idx]
    # Second order theta in square matrix form
    theta2 = numpy.zeros([N, N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    # Calculate unnormalized probabilities for message computation
    psi_i = numpy.exp(theta1)
    psi_i_ij = numpy.exp(theta1 + theta2)
    # Actual belief propogation algorithm
    messages = propagate_beliefs(psi_i, psi_i_ij, N, alpha)
    # Compute beliefs from messages
    b_i, b_ij = compute_beliefs_BP(messages, theta1, theta2, N)
    # Get eta vector
    eta = numpy.empty(theta.shape)
    eta[:N] = b_i[:,1]
    eta[N:] = b_ij[triu_idx[0],triu_idx[1],3]
    # Psi
    bethe_free = -(numpy.sum(numpy.log(b_ij[triu_idx[0], triu_idx[1], 0])) -
                   numpy.sum(((N-1)-1)*numpy.log(b_i[:, 0])))
    if return_psi:
        return eta, bethe_free
    else:
        return eta


def propagate_beliefs(psi_i, psi_i_ij, N, alpha=.5):
    """ Actual belief propagation algorithm [Yedidia, 2001]

    :param numpy.ndarray psi_i:
        (c,) dimensional array with exp(theta_i)
    :param numpy.ndarray psi_i_ij:
        (c,c) dimensional array with exp(theta_i + theta_ij)
    :param int N:
        Number of cells
    :param float alpha:
        Step size for message update (default=0.5)
    :return
        (c,c,2) dimensional array with messages from cells to each other about
        their states
    """

    # Initialize messages
    messages = numpy.ones([N,N,2])
    # Initialize convergence criteria
    message_difference = numpy.inf
    iter_num = 0

    while message_difference > 1e-15 and iter_num <= 1000:
        # Initialize matrix for updated messages
        new_messages = numpy.ones([N,N,2])
        # Compute log of old messages
        log_messages = numpy.log(messages)
        # Marginalize over message sending neurons
        sum_log_messages = numpy.sum(log_messages, axis=0)
        # Compute new messages for neurons being silent
        new_messages[:,:,0] = psi_i*numpy.exp(
            sum_log_messages[:, 1, numpy.newaxis] - log_messages[:, :, 1].T)\
                                        + numpy.exp(
            sum_log_messages[:, 0, numpy.newaxis] - log_messages[:, :, 0].T)
        # Compute new messages for neurons firing
        new_messages[:,:,1] = psi_i_ij*numpy.exp(
            sum_log_messages[:, 1, numpy.newaxis] - log_messages[:, :, 1].T)\
                                        + numpy.exp(
            sum_log_messages[:, 0, numpy.newaxis]-log_messages[:, :, 0].T)
        # Compute normalization
        k = numpy.sum(new_messages, axis=2)
        new_messages = new_messages/k[:,:,numpy.newaxis]
        # Maximal change in messages
        message_difference = numpy.amax(numpy.absolute(messages - new_messages))
        # Normalize and update messages
        M = (1. - alpha)*messages + alpha*new_messages
        k = numpy.sum(M, axis=2)
        messages = M/k[:, :, numpy.newaxis]
        iter_num += 1
        # Raise exception if not converged
        if iter_num == 1000:
            raise Exception('BP algorithm did not converge!')

    # Return messages
    return messages


def compute_beliefs_BP(messages, theta1, theta2, N, all=True):
    """
    Calculates the beliefs.

    :param numpy.ndarray messages:
        (c,c,2) dimensional array with messages from cells to each other about
        their states
    :param numpy.ndarray theta1:
        (c,c) array with theta_i in rows
    :param numpy.ndarray theta2:
        (c,c) array with theta_ij
    :param int N:
        Number of cells
    :param bool all:
        If true first and second order beliefs are returned. Else only
        first order. (Default=True)

    :return:
        (c,2) array containing the belief that  and (c,c,4) array that for pairs
    """
    b_i = numpy.empty([N, 2])
    # Compute unnormalized first order beliefs
    b_i[:, 1] = numpy.exp(theta1[:, 0])*numpy.prod(messages[:, :, 1], axis=0)
    b_i[:, 0] = numpy.prod(messages[:, :, 0], axis=0)
    # Normalize
    k_i = numpy.sum(b_i, axis=1)
    b_i /= k_i[:,numpy.newaxis]
    # Compute unnormalized pair beliefs for x_i = 1
    if all:
        b_ij = numpy.empty([N,N,4])
        # for x_i = 0
        b_ij[:,:,0] = numpy.prod(messages[:,:,0], axis=0)[:,numpy.newaxis]/\
                      messages[:,:,0].T * numpy.prod(messages[:,:,0],axis=0)[
                                         numpy.newaxis,:]/messages[:,:,0]
        b_ij[:,:,1] = numpy.exp(theta1.T)*numpy.prod(
            messages[:,:,0], axis=0)[:,numpy.newaxis]/messages[:,:,0].T\
                      *numpy.prod(messages[:,:,1],axis=0)[numpy.newaxis,:]/\
                      messages[:,:,1]
        # for x_i = 1
        b_ij[:,:,2] = numpy.exp(theta1) * numpy.prod(messages[:,:,1], axis=0)[
                                         :,numpy.newaxis]/messages[:,:,1].T\
                      * numpy.prod(messages[:,:,0],axis=0)[numpy.newaxis,
                        :]/messages[:,:,0]
        b_ij[:,:,3] = numpy.exp(theta1 + theta1.T + theta2) * numpy.prod(
            messages[:,:,1], axis=0)[:,numpy.newaxis]\
                      /messages[:,:,1].T *numpy.prod(messages[:,:,1],axis=0)[
                                          numpy.newaxis,:]/messages[:,:,1]
        k_ij = numpy.sum(b_ij, axis=2)
        b_ij /= k_ij[:,:,numpy.newaxis]

        return b_i, b_ij
    else:
        return b_i


def log_marginal_BP(emd, period=None):
    """
    Computes the approximate log marginal probability of the observed spike
    pattern rates by Bethe approximation (Belief propagation).

    This is just a wrapper function for `log_marginal_raw_BP`. It unpacks data
    from the EMD container object and calls that function.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """
    # Unwrap the parameters and call the raw function
    log_p = log_marginal_raw_BP(emd.theta_f, emd.theta_o, emd.sigma_f,
                                emd.sigma_o_inv, emd.y, emd.R, emd.N, period)

    return log_p


def log_marginal_raw_BP(theta_f, theta_o, sigma_f, sigma_o_inv, y, R, N,
                        period=None):
    """
    Computes the approximate log marginal probability of the observed spike
    pattern rates by Bethe approximation (Belief propagation).

    From within SSLL, this function should be accessed by calling
    `log_marginal_BP` with the EMD container as a parameter.

    See the container.py for a full description of the parameter properties.

    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """
    if period == None: period = (0, theta_f.shape[0])
    # Initialise
    log_p = 0
    # Iterate over each timestep and compute...
    a, b = 0, 0
    for i in range(period[0], period[1]):
        a += log_likelihood_BP(y[i,:], theta_f[i,:], R, N)
        theta_d = theta_f[i,:] - theta_o[i,:]
        b -= numpy.dot(theta_d, sigma_o_inv[i,:]*theta_d)
        logdet_sigma_f = numpy.sum(numpy.log(sigma_f[i]))
        logdet_sigma_o_inv = numpy.sum(numpy.log(sigma_o_inv[i]))
        b += logdet_sigma_f + logdet_sigma_o_inv
    log_p = a + b / 2

    return log_p


def log_likelihood_BP(y_t, theta_f_t, R, N):
    """ Computes the log-likelihood with Bethe approximation  (Belief
    propagation).

    :param numpy.ndarray y_t:
        Frequency of observed patterns for one timestep.
    :param numpy.ndarray theta_f_t:
        Natural parameters of observed patterns for one timestep.
    :param int R:
        Number of trials over which patterns were observed.
    :param int N:
        Number of cells

    :returns:
        Log likelhood of the observed patterns given the natural parameters,
        as a float.
    """
    psi_bethe = compute_eta_BP(theta_f_t, N, return_psi=1)[1]
    log_p = R * (numpy.dot(y_t, theta_f_t) - psi_bethe)
    return log_p


def log_marginal_hybrid(emd, period=None):
    """
    Computes the approximate log marginal probability of the observed spike
    pattern rates by Bethe approximation (hybrid).

    This is just a wrapper function for `log_marginal_raw_hybrid`. It unpacks
    data from the EMD container object and calls that function.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """
    # Unwrap the parameters and call the raw function
    log_p = log_marginal_raw_hybrid(emd.theta_f, emd.theta_o, emd.sigma_f,
                                    emd.sigma_o_inv, emd.y, emd.R, emd.N,
                                    period)

    return log_p


def log_marginal_raw_hybrid(theta_f, theta_o, sigma_f, sigma_o_inv, y, R, N,
                            period=None):
    """
    Computes the approximate log marginal probability of the observed spike
    pattern rates by Bethe approximation (hybrid).

    From within SSLL, this function should be accessed by calling
    `log_marginal_hybrid` with the EMD container as a parameter.

    See the container.py for a full description of the parameter properties.

    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """
    if period == None: period = (0, theta_f.shape[0])
    # Initialise
    log_p = 0
    # Iterate over each timestep and compute...
    a, b = 0, 0
    for i in range(period[0], period[1]):
        try:
            a += log_likelihood_BP(y[i,:], theta_f[i,:], R, N)
        except:
            a += log_likelihood_CCCP(y[i,:], theta_f[i,:], R, N)
        theta_d = theta_f[i,:] - theta_o[i,:]
        b -= numpy.dot(theta_d, sigma_o_inv[i]*theta_d)
        b += numpy.sum(numpy.log(sigma_f[i])) +\
             numpy.sum(numpy.log(sigma_o_inv[i]))
    log_p = a + b / 2

    return log_p


def compute_eta_hybrid(theta, N, return_psi=False):
    """ Function to get eta from the hybrid method.

    :param numpy.ndarray theta:
        (d,) dimensional vector containing theta parameters
    :param int N:
        Number of cells.
    :param bool return_psi:
        If true approximated psi is returned in addition. (Default=False)
    :return:
        (d,) dimensional vector with approximated eta parameters, and psi if
        indicated before
    """
    try:
        return compute_eta_BP(theta, N, return_psi=return_psi)
    except:
        return compute_eta_CCCP(theta, N, return_psi=return_psi)