"""
Classes for encapsulating data used in the expectation maximisation algorithm.

Changes to the original code to incorporate approximations

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

import transforms
import mean_field
import pseudo_likelihood
import bethe_approximation
import max_posterior
import probability


class EMData:
    """
    Contains all of the data used by the EM algorithm, purely for convenience.
    Takes spike trains as an input and computes the observable spike
    (co)incidences (patterns). Initialises the means and covariances of the
    filtered- smoothed- and one-step-prediction natural-parameter distributions.
    Initialises the autoregressive and state-transition hyperparameters.

    :param numpy.ndarray spikes:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c.
    :param int order:
        Order of spike-train interactions to estimate, for example, 2 =
        pairwise, 3 = triplet-wise...
    :param int window:
        Bin-width for counting spikes, in milliseconds.
    :param str param_est:
        Parameter whether exact likelihood ('exact') or pseudo likelihood
        ('pseudo') should be used
    :param str param_est_eta:
        Eta parameters are either calculated exactly ('exact'), by mean
        field TAP approximation ('TAP'), or Bethe approximation (belief
        propagation-'bethe_BP', CCCP-'bethe_CCCP', hybrid-'bethe_hybrid')
    :param function map_function:
        A function from max_posterior.py or pseudo_likelihood.py
        that returns an estimate of the posterior distribution of natural
        parameters for a given timestep.
    :param float lmbda1:
        Inverse coefficient on the identity matrix of the initial
        state-transition covariance matrix for the first order theta parameters.
    :param float lmbda2:
        Inverse coefficient on the identity matrix of the initial
        state-transition covariance matrix for the second order theta parameters.

    :ivar numpy.ndarray spikes:
        Reference to the input spikes.
    :ivar int order:
        Copy of the `order' parameter.
    :ivar int window:
        Copy of the `window' parameter.
    :ivar function max_posterior:
        A function from max_posterior.py that returns an estimate of the
        posterior distribution of natural parameters for a given timestep.
    :ivar function  marg_llk:
        A function that returns the marginal log-likelihood or pseudo-log-
        likelihood.
    :ivar int T:
        Number of timestep in the pattern-counts; should equal the length of the
        spike trains divided by the window.
    :ivar int R:
        Number of trials in the `spikes' input.
    :ivar int N:
        Number of cells in the `spikes' input.
    :ivar int D:
        Dimensions of the natural-parameter distributions, equal to
        D = sum_{k=1}^{`order'} {`N' \choose k}. Density means are all of
        shape (T, D, 1), covariances are (T, D, D) and hyperparameters are
        (D, D).
    :ivar numpy.ndarray y:
        Mean rates of each spike pattern at each timestep, in a 2D array of
        dimesions (T, D).
    :ivar numpy.ndarray theta_o:
        One-step-prediction density mean. Data at theta_o[0] describes the
        probability of the initial state.
    :ivar numpy.ndarray theta_f:
        Filtered density mean.
    :ivar numpy.ndarray theta_s:
        Smoothed density mean.
    :ivar numpy.ndarray eta:
        Estimates rate (conditional rate for pseudo likelihood).
    :ivar numpy.ndarray sigma_o:
        One-step-prediction density covariance.
    :ivar numpy.ndarray sigma_o_inv:
        Inverse of one-step-prediction density covariance
    :ivar numpy.ndarray sigma_f:
        Filtered density covariance.
    :ivar numpy.ndarray sigma_s:
        Smoothed density covariance.
    :ivar numpy.ndarray sigma_s_lag
        Smoothed density lag-one covariance.
    :ivar numpy.ndarray F:
        Autoregressive parameter of state transitions.
    :ivar numpy.ndarray Q:
        Covariance matrix of state-transition probability.
    :ivar int iterations:
        Number of EM iterations for which the algorithm ran.
    :ivar float convergence:
        Ratio between previous and current log-marginal prob. on last iteration.
    """
    def __init__(self, spikes, order, window, param_est, param_est_eta, map_function,
                 lmbda1, lmbda2):

        # Record the input parameters
        self.spikes, self.order, self.window = spikes, order, window
        T, self.R, self.N = self.spikes.shape
        if param_est == 'exact':
            transforms.initialise(self.N, self.order)
            self.max_posterior = max_posterior.functions[map_function]
        elif param_est == 'pseudo':
            pseudo_likelihood.compute_Fx_s(self.spikes, self.order)
            self.max_posterior = pseudo_likelihood.functions[map_function]

        self.param_est_theta = param_est
        self.param_est_eta = param_est_eta

        self.marg_llk = log_marginal_functions[param_est_eta]
        # Compute the `sample' spike-train interactions from the input spikes
        self.y = transforms.compute_y(self.spikes, self.order, self.window)
        # Count timesteps, trials, cells and interaction dimensions

        self.T, self.D = self.y.shape
        assert self.T == T / window
        # Initialise one-step-prediction- filtered- smoothed-density means
        self.theta_o = numpy.zeros((self.T,self.D))
        self.theta_f = numpy.zeros((self.T,self.D))
        self.theta_s = numpy.zeros((self.T,self.D))


        # Initialise covariances of the same (an I-matrix for each timestep)
        if param_est == 'exact':
            I = [numpy.identity(self.D) for i in range(self.T)]
            I = numpy.vstack(I).reshape((self.T,self.D,self.D))
            self.sigma_o = .1 * I
            self.sigma_o_inv = 1./.1 * I
            del I
            # Intialise autoregressive and transition probability hyperparameters
            self.sigma_f = numpy.copy(self.sigma_o)
            self.sigma_s = numpy.copy(self.sigma_o)
            self.sigma_s_lag = numpy.copy(self.sigma_o)
        # For approximate term initialize only the diagonal of the convariances
        else:
            self.sigma_o = .1*numpy.ones((self.T,self.D))
            self.sigma_o_inv = 1./.1*numpy.ones((self.T,self.D))
            self.sigma_f = .1*numpy.ones((self.T,self.D))
            self.sigma_s = .1*numpy.ones((self.T,self.D))
            self.sigma_s_lag = .1*numpy.ones((self.T,self.D))
        self.F = numpy.identity(self.D)
        self.Q = numpy.zeros([self.D, self.D])
        self.Q[:self.N, :self.N] = 1. / lmbda1 * numpy.identity(self.N)
        self.Q[self.N:, self.N:] = 1. / lmbda2 * numpy.identity(self.D - self.N)
        self.mllk = numpy.inf
        # Metadata about EM algorithm execution
        self.iterations, self.convergence = 0, numpy.inf

# Different approximations of the marginal log likelihood
log_marginal_functions = {'exact': probability.log_marginal,
                          'mf': mean_field.log_marginal,
                          'bethe_BP': bethe_approximation.log_marginal_BP,
                          'bethe_CCCP': bethe_approximation.log_marginal_CCCP,
                          'bethe_hybrid': bethe_approximation.log_marginal_hybrid}