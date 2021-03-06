"""
Minimal working example of the SSLL program, examining second-order interaction
between two cells. Approximate inference is used. Note that modules are
imported at the top of each section that uses them (in contrast to the usual
convention of importing all modules at the top of the file) so as to be
completely explicit about the external requirements.

For example for exact methods see 'example_exact.py'.
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

# Set time (milliseconds), number of trials, and number of cells
T, R, N = 500, 200, 10
# Set the interaction order
O = 2


# ----- SPIKE SYNTHESIS -----
# Global module
import numpy
# Local modules
import synthesis
import transforms

# Create underlying time-varying theta parameters as Gaussian processes
theta = synthesis.generate_thetas(N, O, T)



# Generate spikes by Gibbs sampling!
# Sample steps: How many samples are dropped to make them independent
# Pre_n: Determines how many samples are used for the burn in phase
spikes = synthesis.generate_spikes_gibbs(theta, N, O, R, pre_n=100,
                                         sample_steps=1)
# Generate spikes in parallel
# num_proc: How many processes are started in parallel
# spikes = synthesis.generate_spikes_gibbs_parallel(theta, N, O, R, pre_n=100,
#                                          sample_steps=1, num_proc=4)

# ----- ALGORITHM EXECUTION -----
# Local module
import __init__ # From outside this folder, this would be 'import ssll'

# Run the algorithm with pseudo likelihood, bethe approximation!
emd = __init__.run(spikes, O, map_function='cg', lmbda1=200, lmbda2=200,
                   param_est='pseudo', param_est_eta='bethe_hybrid')

# Run the algorithm with pseudo likelihood, TAP approximation!
# emd = __init__.run(spikes, O, map_function='cg', lmbda1=200, lmbda2=200,
#                    param_est='pseudo', param_est_eta='mf')


# ----- PLOTTING -----
# Global module
import pylab

# Set up an output figure
fig, ax = pylab.subplots(2, 1, sharex=True)
# Plot underlying theta traces
ax[0].plot(theta[:,0], c='b', linestyle='--')
ax[0].plot(theta[:,1], c='r', linestyle='--')
ax[1].plot(theta[:,N], c='g', linestyle='--')

# Plot estimated theta traces
ax[0].plot(emd.theta_s[:,0], c='b')
ax[0].plot(emd.theta_s[:,1], c='r')
ax[1].plot(emd.theta_s[:,N], c='g')

# Set labels
ax[0].set_title('Two cells from $N=10$ population')
ax[0].set_ylabel('First-order theta')
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('Second-order theta')
# Show figure!
pylab.show()
