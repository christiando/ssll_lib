"""
Master file of the State-Space Analysis of Spike Correlations.

TODO some sort of automatic versioning system

Changes: All adjustments to incorporate the approximation methods
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
import pdb

import container
import exp_max
import probability
import max_posterior
import synthesis
import transforms
import pseudo_likelihood
import mean_field
import energies
import bethe_approximation


def run(spikes, order, window=1, map_function='nr', lmbda1=100, lmbda2=200,
        max_iter=100, param_est='exact', param_est_eta='exact'):
    """
    Master-function of the State-Space Analysis of Spike Correlation package.
    Uses the expectation-maximisation algorithm to find the probability
    distributions of natural parameters of spike-train interactions over time.
    Calls slave functions to perform the expectation and maximisation steps
    repeatedly until the data likelihood reaches an asymptotic value.

    Note that the execution of some slave functions to this master function are
    of exponential complexity with respect to the `order' parameter.

    :param numpy.ndarray spikes:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c.
    :param int order:
        Order of spike-train interactions to estimate, for example, 2 =
        pairwise, 3 = triplet-wise...
    :param int window:
        Bin-width for counting spikes, in milliseconds.
    :param string map_function:
        Name of the function to use for maximum a-posterior estimation of the
        natural parameters at each timestep. Refer to max_posterior.py.
    :param float lmbda1:
        Inverse coefficient on the identity matrix of the initial
        state-transition covariance matrix for the first order theta parameters.
    :param float lmbda2:
        Inverse coefficient on the identity matrix of the initial
        state-transition covariance matrix for the second order theta parameters.
    :param int max_iter:
        Maximum number of iterations for which to run the EM algorithm.
    :param str param_est:
        Parameter whether exact likelihood ('exact') or pseudo likelihood
        ('pseudo') should be used
    :param str param_est_eta:
        Eta parameters are either calculated exactly ('exact'), by mean
        field TAP approximation ('TAP'), or Bethe approximation (belief
        propagation-'bethe_BP', CCCP-'bethe_CCCP', hybrid-'bethe_hybrid')

    :returns:
        Results encapsulated in a container.EMData object, containing the
        smoothed posterior probability distributions of the natural parameters
        of the spike-train interactions at each timestep, conditional upon the
        given spikes.
    """
    # Ensure NaNs are caught
    numpy.seterr(invalid='raise')
    # Get Number of cells
    N = spikes.shape[2]
    # Initialise the EM-data container
    emd = container.EMData(spikes, order, window, param_est, param_est_eta, map_function, lmbda1, lmbda2)
    # Solves backward problem. For zero rates in the beginning small number is added
    if emd.order == 2:
        #try:
        y_init = numpy.mean(emd.y, axis=0)
        y_init[y_init == 0] = numpy.spacing(1)
        #    emd.theta_o[0] = mean_field.backward_problem(y_init, emd.N, 'TAP')
        #except numpy.linalg.linalg.LinAlgError:
        emd.theta_o[0][:emd.N] = energies.compute_ind_theta(y_init[:emd.N])

    # Set up loop guards for the EM algorithm
    lmp = -numpy.inf
    lmc = emd.marg_llk(emd)
    # Iterate the EM algorithm until convergence or failure
    while (emd.iterations < max_iter) and (emd.convergence > exp_max.CONVERGED):
        print 'EM Iteration: %d - Convergence %.6f > %.6f' %(emd.iterations,
                                                             emd.convergence,
                                                             exp_max.CONVERGED)
        # Perform EM
        exp_max.e_step(emd)
        exp_max.m_step(emd)
        # Update previous and current log marginal values
        lmp = lmc
        lmc = emd.marg_llk(emd)
        emd.mllk = lmc
        # Update EM algorithm metadata
        emd.iterations += 1
        emd.convergence = numpy.absolute((lmp - lmc) / lmp)
    return emd