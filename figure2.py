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
import h5py
import bethe_approximation, synthesis, transforms, __init__
from matplotlib import pyplot
import __init__ as ssll


def generate_data_figure2(data_path='../Data/', max_network_size=60):
    N, O, R, T = 10, 2, 200, 500
    num_of_networks = max_network_size/N
    mu = numpy.zeros(T)
    x = numpy.arange(1, 401)
    mu[100:] = 1. * (3. / (2. * numpy.pi * (x / 400. * 3.) ** 3)) ** .5 * \
               numpy.exp(-3. * ((x / 400. * 3.) - 1.) ** 2 /
                         (2. * (x / 400. * 3.)))

    D = transforms.compute_D(N, O)
    thetas = numpy.empty([num_of_networks, T, D])
    etas = numpy.empty([num_of_networks, T, D])
    psi = numpy.empty([num_of_networks, T])
    S = numpy.empty([num_of_networks, T])
    C = numpy.empty([num_of_networks, T])
    transforms.initialise(N, O)
    for i in range(num_of_networks):
        thetas[i] = synthesis.generate_thetas(N, O, T, mu1=-2.)
        thetas[i, :, :N] += mu[:, numpy.newaxis]
        for t in range(T):
            p = transforms.compute_p(thetas[i, t])
            etas[i, t] = transforms.compute_eta(p)
            psi[i, t] = transforms.compute_psi(thetas[i, t])
            psi1 = transforms.compute_psi(.999 * thetas[i, t])
            psi2 = transforms.compute_psi(1.001 * thetas[i, t])
            C[i, t] = (psi1 - 2. * psi[i, t] + psi2) / .001 ** 2
            S[i, t] = -(numpy.sum(etas[i, t] * thetas[i, t]) - psi[i, t])
    C /= numpy.log(2)
    S /= numpy.log(2)
    f = h5py.File(data_path + 'figure2data.h5', 'w')
    g1 = f.create_group('data')
    g1.create_dataset('thetas', data=thetas)
    g1.create_dataset('etas', data=etas)
    g1.create_dataset('psi', data=psi)
    g1.create_dataset('S', data=S)
    g1.create_dataset('C', data=C)
    g2 = f.create_group('error')
    g2.create_dataset('MISE_thetas', shape=[num_of_networks])
    g2.create_dataset('MISE_population_rate', shape=[num_of_networks])
    g2.create_dataset('MISE_psi', shape=[num_of_networks])
    g2.create_dataset('MISE_S', shape=[num_of_networks])
    g2.create_dataset('MISE_C', shape=[num_of_networks])
    g2.create_dataset('population_rate', shape=[num_of_networks, T])
    g2.create_dataset('psi', shape=[num_of_networks, T])
    g2.create_dataset('S', shape=[num_of_networks, T])
    g2.create_dataset('C', shape=[num_of_networks, T])
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

        spikes = synthesis.generate_spikes_gibbs_parallel(theta_all
                                                          , (i + 1) * N, O, R,
                                                          sample_steps=10,
                                                          num_proc=4)
        emd = __init__.run(spikes, O, map_function='cg', param_est='pseudo',
                           param_est_eta='bethe_hybrid', lmbda1=100,
                           lmbda2=200)

        eta_est = numpy.empty(emd.theta_s.shape)
        psi_est = numpy.empty(T)
        S_est = numpy.empty(T)
        C_est = numpy.empty(T)
        for t in range(T):
            eta_est[t], psi_est[t] = bethe_approximation.compute_eta_hybrid(
                emd.theta_s[t], (i + 1) * N, return_psi=1)
            psi1 = bethe_approximation.compute_eta_hybrid(
                .999 * emd.theta_s[t], (i + 1) * N, return_psi=1)[1]
            psi2 = bethe_approximation.compute_eta_hybrid(
                1.001 * emd.theta_s[t], (i + 1) * N, return_psi=1)[1]
            S_est[t] = -(numpy.sum(eta_est[t] * emd.theta_s[t]) - psi_est[t])
            C_est[t] = (psi1 - 2. * psi_est[t] + psi2) / .001 ** 2
        S_est /= numpy.log(2)
        C_est /= numpy.log(2)
        population_rate = numpy.mean(numpy.mean(etas[:i + 1, :, :N], axis=0),
                                     axis=1)
        population_rate_est = numpy.mean(eta_est[:, :(i + 1) * N], axis=1)
        psi_true = numpy.sum(psi[:(i + 1), :], axis=0)
        S_true = numpy.sum(S[:(i + 1), :], axis=0)
        C_true = numpy.sum(C[:(i + 1), :], axis=0)

        f = h5py.File(data_path + 'figure2data.h5', 'r+')
        f['error']['MISE_thetas'][i] = numpy.mean(
            (theta_all - emd.theta_s) ** 2)
        f['error']['MISE_population_rate'][i] = numpy.mean(
            (population_rate - population_rate_est) ** 2)
        f['error']['MISE_psi'][i] = numpy.mean((psi_est - psi_true) ** 2)
        f['error']['MISE_S'][i] = numpy.mean((S_est - S_true) ** 2)
        f['error']['MISE_C'][i] = numpy.mean((C_est - C_true) ** 2)
        f['error']['population_rate'][i] = population_rate_est
        f['error']['psi'][i] = psi_est
        f['error']['S'][i] = S_est
        f['error']['C'][i] = C_est
        f.close()

    f = h5py.File(data_path + 'figure2data.h5', 'r+')
    thetas = f['data']['thetas'].value
    etas = f['data']['etas'].value
    psi = f['data']['psi'].value
    S = f['data']['S'].value
    C = f['data']['C'].value

    g2 = f.create_group('error500')
    g2.create_dataset('population_rate', shape=[num_of_networks, T])
    g2.create_dataset('psi', shape=[num_of_networks, T])
    g2.create_dataset('S', shape=[num_of_networks, T])
    g2.create_dataset('C', shape=[num_of_networks, T])
    g2.create_dataset('MISE_thetas', shape=[num_of_networks])
    g2.create_dataset('MISE_population_rate', shape=[num_of_networks])
    g2.create_dataset('MISE_psi', shape=[num_of_networks])
    g2.create_dataset('MISE_S', shape=[num_of_networks])
    g2.create_dataset('MISE_C', shape=[num_of_networks])
    f.close()

    R = 500

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
                                                          num_proc=4)
        emd = __init__.run(spikes, O, map_function='cg', param_est='pseudo',
                           param_est_eta='bethe_hybrid', lmbda1=100,
                           lmbda2=200)

        eta_est = numpy.empty(emd.theta_s.shape)
        psi_est = numpy.empty(T)
        S_est = numpy.empty(T)
        C_est = numpy.empty(T)

        for t in range(T):
            eta_est[t], psi_est[t] = \
                bethe_approximation.compute_eta_hybrid(emd.theta_s[t],
                                                       (i + 1) * N,
                                                       return_psi=1)
            psi1 = bethe_approximation.compute_eta_hybrid(.999 * emd.theta_s[t],
                                                          (i + 1) * N,
                                                          return_psi=1)[1]
            psi2 = bethe_approximation.compute_eta_hybrid(
                1.001 * emd.theta_s[t], (i + 1) * N, return_psi=1)[1]
            S_est[t] = -(numpy.sum(eta_est[t] * emd.theta_s[t]) - psi_est[t])
            C_est[t] = (psi1 - 2. * psi_est[t] + psi2) / .001 ** 2
        S_est /= numpy.log(2)
        C_est /= numpy.log(2)
        population_rate = numpy.mean(numpy.mean(etas[:i + 1, :, :N], axis=0),
                                     axis=1)
        population_rate_est = numpy.mean(eta_est[:, :(i + 1) * N], axis=1)
        psi_true = numpy.sum(psi[:(i + 1), :], axis=0)
        S_true = numpy.sum(S[:(i + 1), :], axis=0)
        C_true = numpy.sum(C[:(i + 1), :], axis=0)

        f = h5py.File(data_path + 'figure2data.h5', 'r+')
        f['error500']['MISE_thetas'][i] = numpy.mean(
            (theta_all - emd.theta_s) ** 2)
        f['error500']['MISE_population_rate'][i] = numpy.mean(
            (population_rate - population_rate_est) ** 2)
        f['error500']['MISE_psi'][i] = numpy.mean((psi_est - psi_true) ** 2)
        f['error500']['MISE_S'][i] = numpy.mean((S_est - S_true) ** 2)
        f['error500']['MISE_C'][i] = numpy.mean((C_est - C_true) ** 2)
        f['error500']['population_rate'][i] = population_rate_est
        f['error500']['psi'][i] = psi_est
        f['error500']['S'][i] = S_est
        f['error500']['C'][i] = C_est
        f.close()


def plot_figure2(data_path='../Data/', plot_path='../Plots/',
                 max_network_size=60):
    f = h5py.File(data_path+'figure2data.h5', 'r')
    psi_color = numpy.array([51, 153., 255]) / 256.
    eta_color = numpy.array([0, 204., 102]) / 256.
    S_color = numpy.array([255, 162, 0]) / 256.
    C_color = numpy.array([204, 60, 60]) / 256.
    num_networks = max_network_size/10
    population_rate = \
        numpy.cumsum(numpy.mean(numpy.mean(
            f['data']['etas'][:, :, :10], axis=1), axis=1))
    population_rate /= numpy.arange(1, num_networks + 1)
    psi = numpy.cumsum(numpy.mean(f['data']['psi'].value, axis=1))
    S = numpy.cumsum(numpy.mean(f['data']['S'].value, axis=1))
    C = numpy.cumsum(numpy.mean(f['data']['C'].value, axis=1))

    theta_error = f['error']['MISE_thetas'].value
    population_rate_error = f['error']['MISE_population_rate'].value
    psi_error = f['error']['MISE_psi'].value
    S_error = f['error']['MISE_S'].value
    C_error = f['error']['MISE_C'].value

    population_rate_mean = numpy.mean(f['error']['population_rate'].value,
                                      axis=1)
    psi_mean = numpy.mean(f['error']['psi'].value, axis=1)
    S_mean = numpy.mean(f['error']['S'].value, axis=1)
    C_mean = numpy.mean(f['error']['C'].value, axis=1)

    theta_error500 = f['error500']['MISE_thetas'].value
    population_rate_error500 = f['error500']['MISE_population_rate'].value
    psi_error500 = f['error500']['MISE_psi'].value
    S_error500 = f['error500']['MISE_S'].value
    C_error500 = f['error500']['MISE_C'].value

    population_rate_mean500 = numpy.mean(f['error500']['population_rate'].value,
                                         axis=1)
    psi_mean500 = numpy.mean(f['error500']['psi'].value, axis=1)
    S_mean500 = numpy.mean(f['error500']['S'].value, axis=1)
    C_mean500 = numpy.mean(f['error500']['C'].value, axis=1)

    N_range = numpy.arange(10, max_network_size+1, 10)
    Ds = numpy.empty(N_range.shape)

    for i, N in enumerate(N_range):
        Ds[i] = ssll.transforms.compute_D(N, 2)

    fig = pyplot.figure(figsize=(14, 6))

    ax = fig.add_subplot(2, 5, 2, aspect='equal')
    ax.set_frame_on(False)
    ax.plot(N_range, population_rate, lw=2, c=[.2, .2, .2])
    ax.plot(N_range, population_rate_mean, lw=3, c=eta_color)
    ax.plot(N_range, population_rate_mean500, lw=3, c=eta_color, ls='--')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(N_range)
    ax.set_yticks([.16, .17, .18])
    ax.set_ylabel('$\\langle p_{\\mathrm{spike}}\\rangle_t$', fontsize=16)

    ax = fig.add_subplot(2, 5, 3, aspect='equal')
    ax.set_frame_on(False)
    ax.plot(N_range, psi, lw=4, c=[.2, .2, .2])
    ax.plot(N_range, psi_mean, lw=3, c=psi_color)
    ax.plot(N_range, psi_mean500, lw=3, c=psi_color, ls='--')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(N_range)
    ax.set_yticks([5., 10.])
    ax.set_ylabel('$\\langle\psi\\rangle_t$', fontsize=16)

    ax = fig.add_subplot(2, 5, 4, aspect='equal')
    ax.set_frame_on(False)
    ax.plot(N_range, S / numpy.log2(numpy.exp(1)), lw=2, c=[.2, .2, .2])
    ax.plot(N_range, S_mean / numpy.log2(numpy.exp(1)), lw=3, c=S_color)
    ax.plot(N_range, S_mean500 / numpy.log2(numpy.exp(1)), lw=3, c=S_color,
            ls='--')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([10., 20., 30.])
    ax.set_xticks(N_range)
    ax.set_ylabel('$\\langle S\\rangle_t$', fontsize=16)

    ax = fig.add_subplot(2, 5, 5, aspect='equal')
    ax.set_frame_on(False)
    ax.plot(N_range, C_mean / numpy.log2(numpy.exp(1)), lw=3, c=C_color)
    ax.plot(N_range, C_mean500 / numpy.log2(numpy.exp(1)), lw=3, c=C_color,
            ls='--')
    ax.plot(N_range, C / numpy.log2(numpy.exp(1)), lw=2, c=[.2, .2, .2])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0., 10., 20.])
    ax.set_xticks(N_range)
    ax.set_ylabel('$\\langle C \\rangle_t$', fontsize=16)

    ax = fig.add_subplot(2, 5, 6, aspect='equal')
    ax.set_frame_on(False)
    ax.plot(N_range, numpy.sqrt(theta_error) * Ds / 100., lw=3, c=[.5, .5, .5])
    ax.plot(N_range, numpy.sqrt(theta_error500) * Ds / 100., lw=3,
            c=[.5, .5, .5], ls='--')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([1, 2, 3])
    ax.set_xticks(N_range)
    ax.set_ylabel('RMSE $\\theta$  $[10^2]$', fontsize=16)

    ax = fig.add_subplot(2, 5, 7, aspect='equal')
    ax.set_frame_on(False)
    ax.plot(N_range, numpy.sqrt(population_rate_error) / population_rate_mean,
            lw=3, c=eta_color)
    ax.plot(N_range,
            numpy.sqrt(population_rate_error500) / population_rate_mean500,
            lw=3, c=eta_color, ls='--')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([.1, .2, .3])
    ax.set_xticks(N_range)
    ax.set_ylabel('Error $p_{\mathrm{spike}}$', fontsize=16)

    ax = fig.add_subplot(2, 5, 8, aspect='equal')
    ax.set_frame_on(False)
    ax.plot(N_range, numpy.sqrt(psi_error) / psi_mean, lw=3, c=psi_color)
    ax.plot(N_range, numpy.sqrt(psi_error500) / psi_mean500, lw=3, c=psi_color,
            ls='--')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([.05, .1, .15])
    ax.set_xticks(N_range)
    ax.set_ylabel('Error $\psi$', fontsize=16)
    ax.set_xlabel('Network size N ', fontsize=16)

    ax = fig.add_subplot(2, 5, 9, aspect='equal')
    ax.set_frame_on(False)
    ax.plot(N_range, numpy.sqrt(S_error) / S_mean / numpy.log2(numpy.exp(1)),
            lw=3, c=S_color)
    ax.plot(N_range, numpy.sqrt(S_error500) / S_mean500 /
            numpy.log2(numpy.exp(1)), lw=3, c=S_color, ls='--')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([.05, .1])
    ax.set_xticks(N_range)
    ax.set_ylabel('Error $S$', fontsize=16)

    ax = fig.add_subplot(2, 5, 10, aspect='equal')
    ax.set_frame_on(False)
    ax.plot(N_range, numpy.sqrt(C_error) / C_mean / numpy.log2(numpy.exp(1)),
            lw=3, c=C_color)
    ax.plot(N_range, numpy.sqrt(C_error500) / C_mean500 /
            numpy.log2(numpy.exp(1)), lw=3, c=C_color, ls='--')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0., .05, .1, ])
    ax.set_xticks(N_range)
    ax.set_ylabel('Error $C$', fontsize=16)
    fig.tight_layout()
    ax = fig.add_axes([0.01, 0.9, .05, .05], frameon=0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(.0, .0, 'A', fontsize=16, fontweight='bold')
    ax = fig.add_axes([0.01, 0.45, .05, .05], frameon=0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(.0, .0, 'B', fontsize=16, fontweight='bold')
    fig.savefig(plot_path+'fig2.eps')
    pyplot.show()

if __name__=='__main__':
    generate_data_figure2(data_path='', max_network_size=20)
    plot_figure2(data_path='',plot_path='', max_network_size=20)