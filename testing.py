"""
Functions for testing results.

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
import pylab
import random
import sys
import unittest
import time

import __init__
import probability
import synthesis
import transforms



def klic(p_theta, q_theta, N):
    """
    Computes the Kullback-Leibler divergence for each timestep of two
    natural-parameter distributions, that is, the measure of information lost in
    the EM estimation of the natural parameters with respect to the actual
    natural parameters.

    TODO explain the use of the eta_map as the interactions vector

    :param numpy.ndarray p_theta:
        Mean of the actual natural parameters.
    :param numpy.ndarray q_theta:
        Mean of the estimated natural parameters.
    :param int N:
        Number of cells from which the natural parameters were generated.
    :param int O:
        Order of interactions observed.

    :returns:
        Kullback-Leibler divergence for each timestep as a numpy.ndarray.
    """
    # Get metadata and patterns
    T, D = p_theta.shape
    fx = transforms.enumerate_patterns(N)
    # Compute divergence for each timestep
    kld = numpy.zeros(T)
    for i in xrange(T):
        # Compute normalisations for current timestep
        phi_q = transforms.compute_psi(q_theta[i,:])
        phi_p = transforms.compute_psi(p_theta[i,:])
        # Compute log probability for each pattern
        log_prob_q = numpy.array(transforms.p_map.dot(q_theta[i,:]) - phi_q)
        log_prob_p = numpy.array(transforms.p_map.dot(p_theta[i,:]) - phi_p)
        # Take the KLD for this timestep
        kld[i] = numpy.sum(numpy.exp(log_prob_q) * log_prob_q -\
                           numpy.exp(log_prob_q) * log_prob_p)

    return kld



class TestEstimator(unittest.TestCase):

    def setUp(self):
        self.T = 500
        self.R = 50
        self.theta_base = -3.
        self.spike_seed = 1
        self.wave_seed = 1


    def plot(self, theta_a, theta_e, sigma_e, y, klic, N, T, D):
        # Set up an output figure
        fig, ax = pylab.subplots(3, 1)
        colours = ['b', 'g', 'r', 'c', 'm', 'y']
        # Plot smoothed densities
        for i in xrange(D):
            # Plot original theta values
            ax[0].plot(theta_a[:, i], ls='--', c=colours[i%len(colours)])
            # Plot data-estimated theta values
            ax[0].plot(theta_e[:,i], ls='-', c=colours[i%len(colours)])
            # Plot data-estimated confidence intervals
            ax[0].fill_between(numpy.arange(T),
                theta_e[:,i] - 2*numpy.sqrt(sigma_e[:,i,i]),
                theta_e[:,i] + 2*numpy.sqrt(sigma_e[:,i,i]),
                color=colours[i%len(colours)], alpha=.25)
            ax[1].plot(y[:,i], c=colours[i%len(colours)], ls='-')
        # Set axes labels and legends
        ax[0].set_ylabel('Theta')
        ax[0].set_title('Smoothed densities')
        ax[1].set_title('Observed pattern rates')
        ax[1].set_ylabel('Rate (patterns/second)')
        ax[2].set_title('KL Divergence')
        ax[2].set_ylabel('Bits')
        ax[2].plot(klic)
        ax[2].set_xlabel('Time (ms)')
        fig.tight_layout()
        pylab.show()


    def run_ssll(self, theta, N, O, map_fun='nr', param_est_val='exact', param_est_eta='exact'):
        # Initialise the library for computing pattern probabilities
        transforms.initialise(N, O)
        # Compute probability from theta values
        p = numpy.zeros((self.T, 2**N))
        for i in xrange(self.T):
            p[i,:] = transforms.compute_p(theta[i,:])
        # Generate spikes according to those probabilities
        spikes = synthesis.generate_spikes(p, self.R, seed=self.spike_seed)
        # Run the algorithm!
        emd = __init__.run(spikes, O, map_function=map_fun,
                           param_est=param_est_val, param_est_eta=param_est_eta)
        # Compute the KL divergence between real and estimated parameters
        kld = klic(theta, emd.theta_s, emd.N)
        # Check that KL divergence is OK
        if numpy.any(kld[50:-50] > .05): # change from .01
            self.plot(theta, emd.theta_s, emd.sigma_s, emd.y, kld, emd.N, emd.T,
                emd.D)
        self.assertFalse(numpy.any(kld[50:-50] > .05)) # change from .01


    def test_1_fo_varying(self):
        print "Test First-Order Time-Varying Interactions."
        # Repeat test for different numbers of neurons
        for N in 2**numpy.arange(4):
            print N
            # Create a regular set of theta parameters for each timestep
            theta = numpy.ones((self.T, N)) * self.theta_base
            # Add time-varying components for some neurons
            numpy.random.seed(self.wave_seed)
            n_random = numpy.random.randint(0, N + 1)
            cells = random.sample(numpy.arange(N), n_random)
            for i in xrange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = cells[i]
                theta[:,idx] = self.theta_base + \
                    self.wave(A, f, phi, self.T * 1e-3)
            # Run the actual test
            self.run_ssll(theta, N, 1)


    def test_2_so_variable(self):
        print "Test Second-Order Time-Varying Interactions."
        # Repeat test for different numbers of neurons
        for N in 2**numpy.arange(1, 4):
            print N
            # Compute dimensionality of natural-parameter distribution
            D = transforms.compute_D(N, 2)
            # Create a regular set of theta parameters for each timestep
            theta = numpy.zeros((self.T, D))
            theta[:,:N] = self.theta_base
            theta[:,N:] = -1.
            # Add time-varying components for some neurons
            numpy.random.seed(self.wave_seed)
            n_random = numpy.random.randint(0, N / 2)
            cells = random.sample(numpy.arange(N), n_random)
            for i in xrange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = cells[i]
                theta[:,idx] = self.theta_base + \
                    self.wave(A, f, phi, self.T * 1e-3)
            # Add time-varying components for some interactions
            n_random = numpy.random.randint(0, D - N)
            interactions = random.sample(numpy.arange(N, D), n_random)
            for i in xrange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(1, 2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = interactions[i]
                theta[:,idx] = self.wave(A, f, phi, self.T * 1e-3)
            # Run the actual test
            self.run_ssll(theta, N, 2)


    def test_3_to_variable(self):
        print "Test Third-Order Time-Varying Interactions."
        # Repeat test for different numbers of neurons
        for N in 2**numpy.arange(2, 4):
            print N
            # Compute dimensionality of natural-parameter distribution
            D = transforms.compute_D(N, 3)
            # Create a regular set of theta parameters for each timestep
            theta = numpy.zeros((self.T, D))
            theta[:,:N] = self.theta_base
            theta[:,N:] = -1.
            # Add time-varying components for some neurons
            numpy.random.seed(self.wave_seed)
            n_random = numpy.random.randint(0, N / 2)
            cells = random.sample(numpy.arange(N), n_random)
            for i in xrange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = cells[i]
                theta[:,idx] = self.theta_base + \
                    self.wave(A, f, phi, self.T * 1e-3)
            # Add time-varying components for some interactions
            n_random = numpy.random.randint(0, D - N)
            interactions = random.sample(numpy.arange(N, D), n_random)
            for i in xrange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(1, 2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = interactions[i]
                theta[:,idx] = self.wave(A, f, phi, self.T * 1e-3)
            # Run the actual test
            self.run_ssll(theta, N, 3)


    def test_4_so_variable_gradient_methods(self):
        print "Test Gradient Algorithms (N=8, O=2, Time-Varying Interactions)."
        # Repeat test for different numbers of neurons
        N, O = 8, 2
        # Create time-varying theta parameters
        theta = synthesis.generate_thetas(N, O, self.T)
        # Run the algorithm!
        # Conjugate Gradient
        tc = time.time()
        self.run_ssll(theta, N, O, map_fun='cg')
        print('cg in %f s' %(time.time() - tc))
        # BFGS
        tc = time.time()
        self.run_ssll(theta, N, O, map_fun='bf')
        print('bfgs in %f s' %(time.time() - tc))


    def test_5_so_variable_pseudolikelihood(self):
        print "Test Psuedolikelihood Algorithm (N=8, O=2, Time-Varying Interactions)."
        # Repeat test for different numbers of neurons
        N, O = 8, 2
        # Create time-varying theta parameters
        theta = synthesis.generate_thetas(N, O, self.T)
        # Run the algorithm!
        # Conjugate Gradient
        tc = time.time()
        self.run_ssll(theta, N, O, map_fun='cg', param_est_val='pseudo', param_est_eta='bethe_hybrid')
        print('cg in %f s' %(time.time() - tc))
        # BFGS
        tc = time.time()
        self.run_ssll(theta, N, O, map_fun='bf', param_est_val='pseudo', param_est_eta='bethe_hybrid')
        print('bfgs in %f s' %(time.time() - tc))

    def wave(self, A, f, phi, T):
        rng = numpy.arange(0, T, 1e-3)
        wave = A * numpy.sin(2 * numpy.pi * f * rng + phi)

        return wave


if __name__ == '__main__':
    unittest.main()
