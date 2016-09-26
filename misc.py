"""
Code for generating Figure 2 of <>.

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
import h5py
import bethe_approximation, synthesis, transforms, __init__
from matplotlib import pyplot
import __init__ as ssll
import time

def generate_data_ctime(data_path='../Data/', max_network_size=60,
                              num_procs=4):
    N, O, R, T = 10, 2, 200, 500
    num_of_networks = max_network_size/N
    mu = numpy.zeros(T)
    x = numpy.arange(1, 401)
    mu[100:] = 1. * (3. / (2. * numpy.pi * (x / 400. * 3.) ** 3)) ** .5 * \
               numpy.exp(-3. * ((x / 400. * 3.) - 1.) ** 2 /
                         (2. * (x / 400. * 3.)))

    D = transforms.compute_D(N, O)
    thetas = numpy.empty([num_of_networks, T, D])
    transforms.initialise(N, O)
    for i in range(num_of_networks):
        thetas[i] = synthesis.generate_thetas(N, O, T, mu1=-2.)
        thetas[i, :, :N] += mu[:, numpy.newaxis]

    R = 500
    f = h5py.File(data_path + 'comp_time_data.h5', 'w')
    f.create_dataset('N', data=numpy.arange(N, max_network_size+N, N))
    f.create_dataset('ctime', shape=[2,num_of_networks])
    f.close()
    for i in range(num_of_networks):
        print 'N=%d' % ((i + 1) * N)
        D = transforms.compute_D((i + 1) * N, O)
        theta_all = numpy.empty([T, D])
        triu_idx = numpy.triu_indices(N, k=1)
        triu_idx_all = numpy.triu_indices((i + 1) * N, k=1)

        for j in range(i + 1):
            theta_all[:, N * j:(j + 1) * N] = thetas[j, :, :N]

        for t in range(T):
            theta_ij = numpy.zeros([(i + 1) * N, (i + 1) * N])
            for j in range(i + 1):
                theta_ij[triu_idx[0] + j * N, triu_idx[1] + j * N] = \
                    thetas[j, t, N:]

            theta_all[t, (i + 1) * N:] = theta_ij[triu_idx_all]

        spikes = synthesis.generate_spikes_gibbs_parallel(theta_all,
                                                          (i + 1) * N, O, R,
                                                          sample_steps=10,
                                                          num_proc=num_procs)
        t1 = time.time()
        result = __init__.run(spikes, O, map_function='cg',
                                    param_est='pseudo',
                           param_est_eta='bethe_hybrid',
                            lmbda1=100,
                           lmbda2=200)
        t2 = time.time()
        ctime_bethe = t2 - t1

        f = h5py.File(data_path + 'comp_time_data.h5', 'r+')
        f['ctime'][0, i] = ctime_bethe
        f.close()

        try:
            t1 = time.time()
            result = __init__.run(spikes, O, map_function='cg',
                                                   param_est='pseudo',
                                                   param_est_eta='mf',
                                                   lmbda1=100,
                                                   lmbda2=200)
            t2 = time.time()
            ctime_TAP = t2 - t1
        except Exception:
            ctime_TAP = numpy.nan

        f = h5py.File(data_path + 'comp_time_data.h5', 'r+')
        f['ctime'][1, i] = ctime_TAP
        f.close()


if __name__=='__main__':
    generate_data_ctime(data_path='', max_network_size=10,
                              num_procs=4)

