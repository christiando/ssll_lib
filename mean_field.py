"""
This file contains all methods that are concerned with the TAP approximation.

---

This code implements approximate inference methods for State-Space Analysis of
Spike Correlations (Shimazaki et al. PLoS Comp Bio 2012). It is an extension of
the existing code from repository <https://github.com/tomxsharp/ssll> (For
Matlab Code refer to <http://github.com/shimazaki/dynamic_corr>). We
acknowledge Thomas Sharp for providing the code for exact inference.

In this library are additional methods provided to perform the State-Space
Analysis approximately. This includes pseudolikelihood, TAP, and Bethe
approximations. For details see: <>

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
import itertools
from scipy.optimize import fsolve


def self_consistent_eq(eta, theta1, theta2, expansion='TAP'):
    """ Generates self-consistent equations for forward problem.

    :param numpy.ndarray eta:
        (c,) vector with individual rates for each cell
    :param numpy.ndarray theta1:
        (c,) vector with first order thetas
    :param numpy.ndarray theta2:
        (c, c) array with second order thetas (theta_ij in row i and column j)
    :param str expansion:
        String that indicates order of approximantion. 'naive' for naive
        mean field and 'TAP' for second order approximation with Osanger
        correction. (default='TAP')

    :returns:
        list of c equations that have to be solved for getting the first order
        etas.
    """
    # TAP equations
    if expansion == 'TAP':
        equations = numpy.log(eta) - numpy.log(1 - eta) - theta1 - \
                    numpy.dot(theta2, eta) - \
                    .5*numpy.dot((.5 - eta)[:,numpy.newaxis]*theta2**2,
                                 (eta - eta**2))
    # Naive Mean field equations
    elif expansion == 'naive':
        equations = numpy.log(eta)- numpy.log(1 - eta) - theta1 - \
                    numpy.dot(theta2, eta)

    return equations


def self_consistent_eq_Hinv(eta, theta1, theta2, expansion='TAP'):
    """ Generates self-consistent equations for forward problem.

    :param numpy.ndarray eta:
        (c,) vector with individual rates for each cell
    :param numpy.ndarray theta1:
        (c,) vector with first order thetas
    :param numpy.ndarray theta2:
        (c, c) array with second order thetas (theta_ij in row i and column j)
    :param str expansion:
        String that indicates order of approximantion. 'naive' for naive mean
        field and 'TAP' for second order approximation with Osanger correction.
        (Default='TAP')

    :returns:
        list of c equations that have to be solved for getting the first order
        etas.
    """
    # TAP equations
    if expansion == 'TAP':
        H_diag =  1./eta + 1./(1 - eta) + .5*numpy.dot(theta2**2,
                                                       (eta - eta**2))
    # Naive Mean field equations
    elif expansion == 'naive':
        H_diag =  1./eta + 1./(1 - eta)
    Hinv = numpy.diag(1./H_diag)
    return Hinv


def forward_problem_hessian(theta, N):
    """ Gets the etas for given thetas. Here a costum-made iterative solver is
    used.

    :param numpy.ndarray theta:
        (d,)-dimensional array containing all thetas
    :param int N:
        Number of cells

    :returns:
        (d,) numpy.ndarray with all etas.
    """
    # Initialize eta vector
    eta = numpy.empty(theta.shape)
    eta_max = 0.5*numpy.ones(N)
    # Extract first order thetas
    theta1 = theta[:N]
    # Get indices
    triu_idx = numpy.triu_indices(N, k=1)
    diag_idx = numpy.diag_indices(N)
    # Write second order thetas into matrix
    theta2 = numpy.zeros([N, N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    conv = numpy.inf
    # Solve self-consistent equations and calculate approximation of
    # fisher matrix
    iter_num = 0
    while conv > 1e-4 and iter_num < 500:
        deta = self_consistent_eq(eta_max, theta1=theta1, theta2=theta2,
                                  expansion='TAP')
        Hinv = self_consistent_eq_Hinv(eta_max, theta1=theta1, theta2=theta2,
                                       expansion='TAP')
        eta_max -= .1*numpy.dot(Hinv, deta)
        conv = numpy.amax(numpy.absolute(deta))
        iter_num += 1
        eta_max[eta_max <= 0.] = numpy.spacing(1)
        eta_max[eta_max >= 1.] = 1. - numpy.spacing(1)
        if iter_num == 500:
            raise Exception('Self consistent equations could not be solved!')

    G_inv = - theta2 - theta2**2*numpy.outer(0.5 - eta_max[:N],
                                             0.5 - eta_max[:N])
    G_inv[diag_idx] = 1./eta_max + 1./(1.-eta_max) + .5*numpy.dot(theta2**2,
                                                                  (eta_max -
                                                                   eta_max**2))
    G = numpy.linalg.inv(G_inv)
    # Compute second order eta
    eta2 = G + numpy.outer(eta_max[:N], eta_max[:N])
    eta[N:] = eta2[triu_idx]
    eta[:N] = eta_max
    eta[eta < 0.] = numpy.spacing(1)
    eta[eta > 1.] = 1. - numpy.spacing(1)
    return eta


def forward_problem(theta, N, expansion):
    """ Gets the etas for given thetas.

    :param numpy.ndarray theta:
        (d,)-dimensional array containing all thetas
    :param int N:
        Number of cells
    :param str expansion:
        String that indicates order of approximantion. 'naive' for naive mean
        field and 'TAP' for second order approximation with Osanger correction.

    :returns:
        (d,) numpy.ndarray with all etas.
    """
    # Initialize eta vector
    eta = numpy.empty(theta.shape)
    # Extract first order thetas
    theta1 = theta[:N]
    # Get indices
    triu_idx = numpy.triu_indices(N, k=1)
    diag_idx = numpy.diag_indices(N)
    # Write second order thetas into matrix
    theta2 = numpy.zeros([N, N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    # Solve self-consistent equations and calculate approximation of
    # fisher matrix
    if expansion == 'TAP':
        f = lambda x: self_consistent_eq(x, theta1=theta1, theta2=theta2,
                                         expansion='TAP')
        try:
            eta[:N] = fsolve(f, 0.1*numpy.ones(N))
        except Warning:
            raise Exception('scipy.fsolve did not compute reliable result!')
        G_inv = - theta2 - theta2**2*numpy.outer(0.5 - eta[:N], 0.5 - eta[:N])
    elif expansion == 'naive':
        f = lambda x: self_consistent_eq(x, theta1=theta1, theta2=theta2,
                                         expansion='naive')
        try:
            eta[:N] = fsolve(f, 0.1*numpy.ones(N))
        except Warning:
            raise Exception('scipy.fsolve did not compute reliable result!')
        G_inv = - theta2

    # Compute Inverse of Fisher
    G_inv[diag_idx] = 1./(eta[:N]*(1-eta[:N]))
    G = numpy.linalg.inv(G_inv)
    # Compute second order eta
    eta2 = G + numpy.outer(eta[:N], eta[:N])
    eta[N:] = eta2[triu_idx]
    return eta


def backward_problem(y_t, N, expansion, diag_weight_trick=True):
    """ Calculates thetas for given etas.

    :param numpy.ndarray y_t:
        (d,) dimensional vector containing rates
    :param numpy.ndarray X_t:
        (t,r) dimesional binary array with spikes
    :param int R:
        Number of trials
    :param str expansion:
        String that indicates order of approximantion. 'naive' for naive mean
        field and 'TAP' for second order approximation with Osanger correction.

    """


    # Compute indices
    triu_idx = numpy.triu_indices(N, k=1)
    diag_idx = numpy.diag_indices(N)
    # Compute covariance matrix and invert
    G = compute_fisher_info_from_eta(y_t, N)
    G_inv = numpy.linalg.inv(G[:N,:N])

    # Solve backward problem for indicated approximation
    if expansion == 'TAP':
        # Write the rate into a matrix for dot-products
        y_mat = numpy.zeros([N, N])
        y_mat[triu_idx] = y_t[N:]
        y_mat[triu_idx[1],triu_idx[0]] = y_t[N:]
        # Compute quadratic coefficient of the solution for theta_ij
        quadratic_term = ((.5 - y_mat)*(.5 - y_mat.T)).flatten()
        # Compute linear coefficient of the solution for theta_ij
        linear_term = numpy.ones(quadratic_term.shape, dtype=float)
        # Compute offset of the solution for theta_ijtheta_TAP_wD
        offset = G_inv.flatten()
        # Solve for theta_ij
        theta2_solution = solve_quadratic_problem(quadratic_term, linear_term,
                                                  offset)
        # Bring back to matrix form
        theta2_est = theta2_solution.reshape([N, N])
        theta2_est[diag_idx] = 0
        # Calculate Diagonal
        if diag_weight_trick:
            theta2_est[diag_idx] = compute_diagonal(y_t[:N], theta2_est,
                                                    G_inv[diag_idx])
        # Initialize array for solution of theta
        theta = numpy.empty(y_t.shape)
        # Fill in theta_ij
        diag_weight = numpy.ones(theta2_est.shape)
        theta[N:] = theta2_est[triu_idx]
        # Compute theta_i
        theta[:N] = numpy.log(y_t[:N]/(1 - y_t[:N])) - \
                        numpy.dot(theta2_est, y_t[:N]) - \
                        0.5*(0.5-y_t[:N])*\
                        numpy.dot(theta2_est**2, y_t[:N]*(1 - y_t[:N]))

    return theta


def compute_diagonal(eta, theta2, G_inv_diag):
    """ Computes the diagonal for the second order theta matrix.

    :param numpy.ndarray eta:
        (c,) vector with all first order rates.
    :param numpy.ndarray theta3:
        (c,c) array with all second order thetas.
    :param G_inv_diag:
        (c,) vector with the diagonal of the Fisher Info.

    :returns:
        (c,) array with solution for theta_ii
    """
    return - 1./(eta*(1 - eta)) - .5*numpy.dot(theta2**2,eta*(1 - eta)) +\
           G_inv_diag


def solve_quadratic_problem(a, b, c):
    """ Solves a quadratic equation of form:

    ax^2 + bx + c = 0

    Selects the solution closest to the naive mean field solution.
    If solution is complex, naive approximation is returned.

    :param numpy.ndarray a:
        d-dimensional vector, where d is the number of equations.
        Vector contains coefficients of quadratic term.
    :param numpy.ndarray b:
        d-dimensional vector, containing coefficients of linear term.
    :param numpy.ndarray c:
        d-dimensional vector, offset.

    :returns:
        x that is closest to naive solution or, if complex, naive mean field
        approx.
    """
    D = a.shape[0]
    # Get solution without quadratic term
    naive_x = -c/b
    # Compute term below root
    term_in_root = b**2 - 4.*a*c
    # Check where solution is non complex
    non_complex = term_in_root >= 0
    non_complex_idx = numpy.where(non_complex)[0]
    is_complex_idx = numpy.where(numpy.logical_not(non_complex))[0]
    # Initialize array for two solutions
    x_12 = numpy.zeros([D, 2])
    # Compute two solutions
    x_12[non_complex_idx, 0] = (-b[non_complex_idx] - \
                                     numpy.sqrt(term_in_root[non_complex_idx]))\
                               /(2.*a[non_complex_idx])
    x_12[non_complex_idx, 1] = (-b[non_complex_idx] + \
                                     numpy.sqrt(term_in_root[non_complex_idx]))\
                               /(2.*a[non_complex_idx])
    # Find closest solution
    diff2naive = numpy.absolute(x_12 - naive_x[:, numpy.newaxis])
    closest_x = numpy.argmin(diff2naive, axis=1)
    sol1 = numpy.where(closest_x == 0)[0]
    sol2 = numpy.where(closest_x)[0]
    x = numpy.zeros(D)
    x[sol1] = x_12[sol1, 0]
    x[sol2] = x_12[sol2, 1]
    # Take naive solution where complex
    x[is_complex_idx] = naive_x[is_complex_idx]
    # Return solution
    return x


def compute_psi(theta, eta, N):
    """ Computes TAP approximation of log-partition function.

    :param numpy.ndarray theta:
        (d,) dimensional vector with natural parameters theta.
    :param numpy.ndarray eta:
        (d,) dimensional vector with expectation parameters eta.
    :param int N:
        Number of cells.

    :returns:
        TAP-approximation of log-partition function
    """
    # Get indices
    triu_idx = numpy.triu_indices(N, k=1)
    # Insert second order theta into matrix
    theta2 = numpy.zeros([N,N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    # Dot product of theta and eta
    psi_trans = numpy.dot(theta[:N], eta[:N])
    # Entropy of independent model
    psi_0 = - numpy.sum(eta[:N]*numpy.log(eta[:N]) + (1 - eta[:N])*
                        numpy.log(1 - eta[:N]))
    # First derivative
    psi_1 = .5*numpy.sum(theta2*numpy.outer(eta[:N],eta[:N]))
    # Second derivative
    psi_2 = .125*numpy.sum(theta2**2*numpy.outer(eta[:N] - eta[:N]**2,
                                                 eta[:N] - eta[:N]**2))
    # Return sum of all
    return psi_trans + psi_0 + psi_1 + psi_2


def log_likelihood_mf(eta, theta, R, N):
    """ Compute log-likelihood with TAP estimation of log-partition function.

    :param numpy.ndarray theta:
        (d,) dimensional vector with natural parameters theta.
    :param eta:
        (d,) dimensional vector with expectation parameters eta.
    :param numpy.ndarray y:
        (d,) dimensional vector with empirical rates.
    :param int N:
        Number of cells.

    """
    # Compute TAP estimation of psi
    th0 = numpy.zeros(theta.shape)
    th0[:N] = theta[:N]
    psi = compute_psi(theta, eta, N)
    # Return log-likelihood
    return R*(numpy.dot(theta, eta) - psi)


def log_marginal(emd, period=None):
    """
    Computes the log marginal probability of the observed spike-pattern rates
    by marginalising over the natural-parameter distributions. See equation 45
    of the source paper for details.

    This is just a wrapper function for `log_marginal_raw`. It unpacks data
    from the EMD container pbject and calls that function.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """
    # Unwrap the parameters and call the raw function
    log_p = log_marginal_raw(emd.theta_f, emd.theta_o, emd.sigma_f,
                             emd.sigma_o_inv, emd.y, emd.R, emd.N, period)

    return log_p


def log_marginal_raw(theta_f, theta_o, sigma_f, sigma_o_inv, y, R, N,
                     period=None):
    """
    Computes the log marginal probability of the observed spike-pattern rates
    by marginalising over the natural-parameter distributions. See equation 45
    of the source paper for details.

    From within SSLL, this function should be accessed by calling
    `log_marginal` with the EMD container as a parameter. This raw function is
    designed to be called from outside SSLL, when a complete EMD container
    might not be available.

    See the container.py for a full description of the parameter properties.

    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """

    if period is None:
        period = (0, theta_f.shape[0])
    # Initialise
    log_p = 0
    # Iterate over each timestep and compute...
    a, b = 0, 0
    for i in range(period[0], period[1]):
        a += log_likelihood_mf(y[i], theta_f[i], R, N)
        theta_d = theta_f[i] - theta_o[i]
        b -= numpy.dot(theta_d, sigma_o_inv[i]*theta_d)
        b += numpy.sum(numpy.log(sigma_f[i])) +\
             numpy.sum(numpy.log(sigma_o_inv[i]))
    log_p = a + b / 2

    return log_p


def compute_fisher_info_from_eta(eta, N):
    """ Creates Fisher-Information matrix from eta-vector.

    :param numpy.ndarray eta:
        vector with rates and coincidence rates
    :param int N:
        number of cells

    :returns:
        Fisher-information matrix as numpy.ndarray
    """
    # Initialize matrix for the first part of the fisher matrix
    G1 = numpy.zeros([N, N])
    # Get upper triangle indices
    triu_idx = numpy.triu_indices(N, k=1)
    # Construct first part from eta
    G1[triu_idx] = eta[N:]
    G1 += G1.T
    G1 += numpy.diag(eta[:N])
    # Second part of fisher information matrix
    G2 = numpy.outer(eta[:N], eta[:N])
    # Final matrix
    G = G1 - G2

    return G


def estimate_higher_order_eta(eta, N, order):
    subpops = list(itertools.combinations(range(N), order))
    pairs_in_subpops = []
    for i in subpops:
        pairs_in_subpops.append(list(itertools.combinations(i, 2)))
    pair_array = numpy.array(pairs_in_subpops)
    sub_pops_array = numpy.array(subpops)
    eta2 = numpy.zeros([N,N])
    triu_idx = numpy.triu_indices(N,1)
    eta2[triu_idx] = eta[N:]
    eta2 += eta2.T
    log_eta_a = numpy.sum(numpy.log(eta2[pair_array[:,:,0],pair_array[:,:,1]]),
                          axis=1)
    eta1 = eta[:N]
    log_eta_b = numpy.sum(numpy.log(eta1[sub_pops_array])*(order-2), axis=1)
    return numpy.exp(log_eta_a - log_eta_b)


def compute_higher_order_etas(eta1, theta2, O):
    """ Approximates higher order thetas by mean-field approximation.

    :param numpy.ndarray eta1:
        (c,) vector with first-order rates
    :param numpy.ndarray theta2:
        (d-c,) vector with second order thetas
    :param int O:
        Order for that the rates should be computed.

    :returns:
        numpy.ndarray with approximating of higher order rates
    """

    # Initialize all necessary parameters
    N = eta1.shape[0]
    triu_idx = numpy.triu_indices(N, k=1)
    theta2_mat = numpy.zeros([N,N])
    theta2_mat[triu_idx] = theta2
    theta2_mat += theta2_mat.T
    # Get all subpopulations for that the rates should be computed
    subpopulations = list(itertools.combinations(range(N),O))
    # Get Connections within the subpopulation
    # (PROBABLY MORE ELEGANT WAY POSSIBLE)
    pairs_in_subpopulations = []
    for i in subpopulations:
        pairs_in_subpopulations.append(list(itertools.combinations(i, 2)))
    # Get Indices for the pairs in an NxN matrix
    pair_idx = numpy.ravel_multi_index(numpy.array(pairs_in_subpopulations).T,
                                       [N,N])
    # Compute independent rates of the subpopulations
    ind_rates = numpy.prod(eta1[subpopulations], axis = 1)
    # Compute the terms of the first derivative responsible for pairs within
    # subpopulation
    terms_within_subpopulation = theta2_mat*(1-numpy.outer(eta1, eta1))
    # Extract the values for each pair within each subpopulation and sum over it
    # (Note that 0.5 is dropped because we consider each pair just once!)
    first_div1 = numpy.sum(terms_within_subpopulation.flatten()[pair_idx],
                           axis=0)
    # Product of theta_ij and eta_j
    theta2_eta_j = theta2_mat*eta1[:,numpy.newaxis]
    # Get the eta_i's for each subpopulation
    eta_i = eta1[subpopulations]
    # Get the connections for each neuron in each subpopulation to units outside
    # the population
    theta2_eta_j_pairs =  theta2_eta_j[subpopulations,:]
    # neighbors of subpopulation
    terms_neighboring_subpopulation = theta2_eta_j_pairs*\
                                      eta_i[:,:,numpy.newaxis]
    # Compute first derivative
    first_derivative = (first_div1 +
                        numpy.sum(numpy.sum(terms_neighboring_subpopulation,
                                            axis=2),axis=1))*ind_rates
    # Compute approximation of higher order rates and return
    higher_order_rates = ind_rates + first_derivative
    return higher_order_rates
