"""
Code for generating Figure 1 of <>.

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
import multiprocessing
from functools import partial
import bethe_approximation, synthesis, transforms, __init__
from matplotlib import pyplot
import itertools
from scipy.stats.mstats import mquantiles
import networkx as nx


def get_sampled_eta_psi(i, theta_sampled, N):
    print i
    psi = numpy.empty([100,3])
    eta = numpy.empty([int(N + N*(N-1)/2),100])
    alpha = [.999,1.,1.001]
    for j in range(100):
        for k, a in enumerate(alpha):
            if k == 1:
                eta[:,j], psi[j,k] = bethe_approximation.compute_eta_hybrid(
                    a*theta_sampled[i,:,j], int(N), return_psi=True)
            else:
                psi[j,k] = bethe_approximation.compute_eta_hybrid(
                    a*theta_sampled[i,:,j], int(N), return_psi=True)[1]
    return eta, psi, i


def generate_data_figure1(data_path = '../Data/'):
    N, O, R, T = 15, 2, 200, 500
    mu = numpy.zeros(T)
    x = numpy.arange(1, 401)
    mu[100:] = 1. * (3. / (2. * numpy.pi * (x/400.*3.) ** 3)) ** .5 * \
               numpy.exp(-3. * ((x/400.*3.) - 1.) ** 2 / (2. * (x/400.*3.)))
    theta1 = synthesis.generate_thetas(N, O, T, mu1=-2.)
    theta2 = synthesis.generate_thetas(N, O, T, mu1=-2.)
    theta1[:, :N] += mu[:, numpy.newaxis]
    theta2[:, :N] += mu[:, numpy.newaxis]
    D = transforms.compute_D(N * 2, O)
    theta_all = numpy.empty([T, D])
    theta_all[:, :N] = theta1[:, :N]
    theta_all[:, N:2 * N] = theta2[:, :N]
    triu_idx = numpy.triu_indices(N, k=1)
    triu_idx_all = numpy.triu_indices(2 * N, k=1)
    for t in range(T):
        theta_ij = numpy.zeros([2 * N, 2 * N])
        theta_ij[triu_idx] = theta1[t, N:]
        theta_ij[triu_idx[0] + N, triu_idx[1] + N] = theta2[t, N:]
        theta_all[t, 2 * N:] = theta_ij[triu_idx_all]

    psi1 = numpy.empty([T, 3])
    psi2 = numpy.empty([T, 3])
    eta1 = numpy.empty(theta1.shape)
    eta2 = numpy.empty(theta2.shape)
    alpha = [.999,1.,1.001]
    transforms.initialise(N, O)
    for i in range(T):
        for j, a in enumerate(alpha):
            psi1[i, j] = transforms.compute_psi(a * theta1[i])
        p = transforms.compute_p(theta1[i])
        eta1[i] = transforms.compute_eta(p)
        for j, a in enumerate(alpha):
            psi2[i, j] = transforms.compute_psi(a * theta2[i])
        p = transforms.compute_p(theta2[i])
        eta2[i] = transforms.compute_eta(p)

    psi_all = psi1 + psi2
    S1 = -numpy.sum(eta1 * theta1, axis=1) + psi1[:, 1]
    S1 /= numpy.log(2)
    S2 = -numpy.sum(eta2 * theta2, axis=1) + psi2[:, 1]
    S2 /= numpy.log(2)
    S_all = S1 + S2

    C1 = (psi1[:, 0] - 2. * psi1[:, 1] + psi1[:, 2]) / .001 ** 2
    C1 /= numpy.log(2)
    C2 = (psi2[:, 0] - 2. * psi2[:, 1] + psi2[:, 2]) / .001 ** 2
    C2 /= numpy.log(2)

    C_all = C1 + C2

    spikes = synthesis.generate_spikes_gibbs_parallel(theta_all, 2 * N, O, R,
                                                      sample_steps=10,
                                                      num_proc=4)

    print 'Model and Data generated'

    emd = __init__.run(spikes, O, map_function='cg', param_est='pseudo',
                       param_est_eta='bethe_hybrid', lmbda1=100, lmbda2=200)

    f = h5py.File(data_path + 'figure1data.h5', 'w')
    g_data = f.create_group('data')
    g_data.create_dataset('theta_all', data=theta_all)
    g_data.create_dataset('psi_all', data=psi_all)
    g_data.create_dataset('S_all', data=S_all)
    g_data.create_dataset('C_all', data=C_all)
    g_data.create_dataset('spikes', data=spikes)
    g_data.create_dataset('theta1', data=theta1)
    g_data.create_dataset('theta2', data=theta2)
    g_data.create_dataset('psi1', data=psi1)
    g_data.create_dataset('S1', data=S1)
    g_data.create_dataset('C1', data=C1)
    g_data.create_dataset('psi2', data=psi2)
    g_data.create_dataset('S2', data=S2)
    g_data.create_dataset('C2', data=C2)
    g_fit = f.create_group('fit')
    g_fit.create_dataset('theta_s', data=emd.theta_s)
    g_fit.create_dataset('sigma_s', data=emd.sigma_s)
    g_fit.create_dataset('Q', data=emd.Q)
    f.close()

    print 'Fit and saved'

    f = h5py.File(data_path + 'figure1data.h5', 'r+')
    g_fit = f['fit']
    theta = g_fit['theta_s'].value
    sigma = g_fit['sigma_s'].value

    X = numpy.random.randn(theta.shape[0], theta.shape[1], 100)
    theta_sampled = \
        theta[:, :, numpy.newaxis] + X * numpy.sqrt(sigma)[:, :, numpy.newaxis]

    T = range(theta.shape[0])
    eta_sampled = numpy.empty([theta.shape[0], theta.shape[1], 100])
    psi_sampled = numpy.empty([theta.shape[0], 100, 3])

    func = partial(get_sampled_eta_psi, theta_sampled=theta_sampled, N=2*N)
    pool = multiprocessing.Pool(10)
    results = pool.map(func, T)

    for eta, psi, i in results:
        eta_sampled[i] = eta
        psi_sampled[i] = psi
    S_sampled = \
        -(numpy.sum(eta_sampled*theta_sampled, axis=1) - psi_sampled[:, :, 1])
    S_sampled /= numpy.log(2)
    C_sampled = \
        (psi_sampled[:, :, 0] - 2.*psi_sampled[:, :, 1] +
         psi_sampled[:, :, 2])/.001**2
    C_sampled /= numpy.log(2)
    g_sampled = f.create_group('sampled_results')
    g_sampled.create_dataset('theta_sampled', data=theta_sampled)
    g_sampled.create_dataset('eta_sampled', data=eta_sampled)
    g_sampled.create_dataset('psi_sampled', data=psi_sampled)
    g_sampled.create_dataset('S_sampled', data=S_sampled)
    g_sampled.create_dataset('C_sampled', data=C_sampled)
    f.close()

    print 'Done'

def plot_figure1(data_path='../Data/', plot_path='../Plots/'):

    N, O = 30, 2
    f = h5py.File(data_path + 'figure1data.h5', 'r')
    # Figure A
    fig = pyplot.figure(figsize=(30, 20))

    ax = fig.add_axes([0.07, 0.68, .4, .4])
    ax.imshow(-f['data']['spikes'][:, 0, :].T, cmap='gray', aspect=5,
              interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_axes([.06, 0.65, .4, .4])
    ax.imshow(-f['data']['spikes'][:, 1, :].T, cmap='gray', aspect=5,
              interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_axes([.05, 0.62, .4, .4])
    ax.imshow(-f['data']['spikes'][:, 2, :].T, cmap='gray', aspect=5,
              interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Time [AU]', fontsize=26)
    ax.set_ylabel('Neuron ID', fontsize=26)

    ax = fig.add_axes([.05, 0.5, .4, .2])
    ax.set_frame_on(False)
    ax.plot(numpy.mean(numpy.mean(f['data']['spikes'][:, :, :], axis=1),
                       axis=1), linewidth=4, color='k')
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([.1, .2, .3])
    ax.set_xticks([50, 150, 300])
    ax.set_ylabel('Data $p_{\\mathrm{spike}}$', fontsize=26)
    ax.set_xlabel('Time [AU]', fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=20)

    theta = f['fit']['theta_s'].value
    sigma_s = f['fit']['sigma_s'].value
    bounds = numpy.empty([theta.shape[0], theta.shape[1] - N, 2])
    bounds[:, :, 0] = theta[:, N:] - 2.58 * numpy.sqrt(sigma_s[:, N:])
    bounds[:, :, 1] = theta[:, N:] + 2.58 * numpy.sqrt(sigma_s[:, N:])
    # Figure B (Networks)
    graph_ax = [fig.add_axes([.52, 0.78, .15, .2]),
                fig.add_axes([.67, 0.78, .15, .2]),
                fig.add_axes([.82, 0.78, .15, .2])]
    T = [50, 150, 300]
    for i, t in enumerate(T):
        idx = numpy.where(numpy.logical_or(bounds[t, :, 0] > 0, bounds[t, :, 1]
                                           < 0))[0]
        conn_idx_all = numpy.arange(0, N * (N - 1) / 2)
        conn_idx = conn_idx_all[idx]
        all_conns = itertools.combinations(range(N), 2)
        conns = numpy.array(list(all_conns))[conn_idx]
        G1 = nx.Graph()
        G1.add_nodes_from(range(N))
        # conns = itertools.combinations(range(30),2)
        G1.add_edges_from(conns)
        pos1 = nx.circular_layout(G1)
        net_nodes = \
            nx.draw_networkx_nodes(G1, pos1, ax=graph_ax[i],
                                   node_color=theta[t, :N],
                                   cmap=pyplot.get_cmap('hot'), vmin=-3,
                                   vmax=-1.)
        e1 = nx.draw_networkx_edges(G1, pos1, ax=graph_ax[i],
                                    edge_color=theta[t, conn_idx].tolist(),
                                    edge_cmap=pyplot.get_cmap('seismic'),
                                    edge_vmin=-.7, edge_vmax=.7, width=2)
        graph_ax[i].axis('off')
        x0, x1 = graph_ax[i].get_xlim()
        y0, y1 = graph_ax[i].get_ylim()
        graph_ax[i].set_aspect(abs(x1 - x0) / abs(y1 - y0))
        graph_ax[i].set_title('t=%d' % t, fontsize=24)
    cbar_ax = fig.add_axes([0.62, 0.79, 0.1, 0.01])
    cbar_ax.tick_params(axis='both', which='major', labelsize=20)
    cbar = fig.colorbar(net_nodes, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-3, -2, -1])
    cbar_ax.set_title('$\\theta_{i}$', fontsize=22)
    cbar_ax = fig.add_axes([0.77, 0.79, 0.1, 0.01])
    cbar = fig.colorbar(e1, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-.5, 0., .5])
    cbar_ax.set_title('$\\theta_{ij}$', fontsize=22)
    cbar_ax.tick_params(axis='both', which='major', labelsize=20)

    # Figure B (Thetas)
    theta = f['data']['theta_all'][:, [165, 170, 182]]
    theta_fit = f['fit']['theta_s'][:, [165, 170, 182]]
    sigma_fit = f['fit']['sigma_s'][:, [165, 170, 182]]
    ax1 = fig.add_axes([.55, 0.68, .4, .1])
    ax1.set_frame_on(False)
    ax1.fill_between(range(0, 500), theta_fit[:, 0] - 2.58 *
                     numpy.sqrt(sigma_fit[:, 0]), theta_fit[:, 0] + 2.58 *
                     numpy.sqrt(sigma_fit[:, 0]), color=[.4, .4, .4])
    ax1.plot(range(500), theta[:, 0], linewidth=4, color='k')
    ax1.set_yticks([-1, 0, 1])
    ax1.set_ylim([-1.1, 1.1])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                 linewidth=2))
    ax1.set_xticks([])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1 = fig.add_axes([.55, 0.57, .4, .1])
    ax1.set_frame_on(False)
    ax1.fill_between(range(0, 500), theta_fit[:, 1] - 2.58 *
                     numpy.sqrt(sigma_fit[:, 1]),
                     theta_fit[:, 1] + 2.58 * numpy.sqrt(sigma_fit[:, 1]),
                     color=[.5, .5, .5])
    ax1.plot(range(500), theta[:, 1], linewidth=4, color='k')
    ax1.set_yticks([-1, 0, 1])
    ax1.set_ylim([-1.1, 1.5])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                 linewidth=2))
    ax1.set_xticks([])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_ylabel('$\\theta_{ij}$', fontsize=26)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1 = fig.add_axes([.55, 0.46, .4, .1])
    ax1.set_frame_on(False)
    ax1.fill_between(range(0, 500),
                     theta_fit[:, 2] - 2.58 * numpy.sqrt(sigma_fit[:, 2]),
                     theta_fit[:, 2] + 2.58 * numpy.sqrt(sigma_fit[:, 2]),
                     color=[.6, .6, .6])
    ax1.plot(range(500), theta[:, 2], linewidth=4, color='k')
    ax1.set_ylim([-1.1, 1.1])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                 linewidth=2))
    ax1.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                 linewidth=3))
    ax1.set_xticks([50, 150, 300])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xlabel('Time [AU]', fontsize=26)
    ax1.set_yticks([-1, 0, 1])
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # Figure C
    psi_color = numpy.array([51, 153., 255]) / 256.
    eta_color = numpy.array([0, 204., 102]) / 256.
    S_color = numpy.array([255, 162, 0]) / 256.
    C_color = numpy.array([204, 60, 60]) / 256.
    psi_quantiles = mquantiles(f['sampled_results']['psi_sampled'][:, :, 1],
                               prob=[.01, .99], axis=1)
    psi_true = f['data']['psi_all'].value
    eta_quantiles = mquantiles(numpy.mean(
        f['sampled_results']['eta_sampled'][:, :N, :], axis=1), prob=[.01, .99],
                               axis=1)
    C_quantiles = mquantiles(f['sampled_results']['C_sampled'][:, :],
                             prob=[.01, .99], axis=1)
    C_true = f['data']['C_all']
    S_quantiles = mquantiles(f['sampled_results']['S_sampled'][:, :],
                             prob=[.01, .99], axis=1)
    S_true = f['data']['S_all']
    eta1 = numpy.empty(f['data']['theta1'].shape)
    eta2 = numpy.empty(f['data']['theta2'].shape)
    T = eta1.shape[0]
    N1, N2 = 15, 15
    transforms.initialise(N1, O)
    for i in range(T):
        p = transforms.compute_p(f['data']['theta1'][i])
        eta1[i] = transforms.compute_eta(p)
        p = transforms.compute_p(f['data']['theta2'][i])
        eta2[i] = transforms.compute_eta(p)

    ax1 = fig.add_axes([.08, 0.23, .4, .15])
    ax1.set_frame_on(False)
    ax1.fill_between(range(0, 500), eta_quantiles[:, 0], eta_quantiles[:, 1],
                     color=eta_color)
    eta_true = 1. / 2. * (numpy.mean(eta1[:, :N1], axis=1) +
                          numpy.mean(eta2[:, :N2], axis=1))
    ax1.fill_between(range(0, 500), eta_quantiles[:, 0], eta_quantiles[:, 1],
                     color=eta_color)
    ax1.plot(range(500), eta_true, linewidth=4, color=eta_color * .8)

    ax1.set_yticks([.1, .2, .3])
    ax1.set_ylim([.09, .35])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                 linewidth=2))
    ax1.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                 linewidth=3))
    ax1.set_xticks([50, 150, 300])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_ylabel('$p_{\\mathrm{spike}}$', fontsize=26)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax1 = fig.add_axes([.08, 0.05, .4, .15])
    ax1.set_frame_on(False)
    ax1.fill_between(range(0, 500), numpy.exp(-psi_quantiles[:, 0]),
                     numpy.exp(-psi_quantiles[:, 1]), color=psi_color)
    ax1.plot(range(500), numpy.exp(-psi_true), linewidth=4, color=psi_color * .8)
    ax1.set_yticks([.0, .01, .02])
    ax1.set_ylim([.0, .025])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                 linewidth=2))
    ax1.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                 linewidth=3))
    ax1.set_xticks([50, 150, 300])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_ylabel('$p_{\\mathrm{silence}}$', fontsize=26)
    ax1.set_xlabel('Time [AU]', fontsize=26)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    # Entropy
    ax2 = fig.add_axes([.52, 0.23, .4, .15])
    ax2.set_frame_on(False)

    ax2.fill_between(range(0, 500), S_quantiles[:, 0] / numpy.log2(numpy.exp(1)),
                     S_quantiles[:, 1] / numpy.log2(numpy.exp(1)), color=S_color)
    ax2.plot(range(500), S_true / numpy.log2(numpy.exp(1)), linewidth=4, color=S_color * .8)
    ax2.set_xticks([50, 150, 300])
    ax2.set_yticks([10, 14, 18])
    ymin, ymax = ax2.get_yaxis().get_view_interval()
    xmin, xmax = ax2.get_xaxis().get_view_interval()
    ax2.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                 linewidth=2))
    ax2.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                 linewidth=3))
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_ylabel('$S$', fontsize=26)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    # Heat capacity
    ax2 = fig.add_axes([.52, 0.05, .4, .15])
    ax2.set_frame_on(False)
    ax2.fill_between(range(0, 500),
                     C_quantiles[:, 0] / numpy.log2(numpy.exp(1)),
                     C_quantiles[:, 1] / numpy.log2(numpy.exp(1)),
                     color=C_color)
    ax2.plot(range(500), C_true / numpy.log2(numpy.exp(1)), linewidth=5,
             color=C_color * .8)
    ymin, ymax = ax2.get_yaxis().get_view_interval()
    xmin, xmax = ax2.get_xaxis().get_view_interval()
    ax2.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                 linewidth=2))
    ax2.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                 linewidth=3))
    ax2.set_xticks([50, 150, 300])
    ax2.set_yticks([5, 10])
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_xlabel('Time [AU]', fontsize=26)
    ax2.set_ylabel('$C$', fontsize=26)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax = fig.add_axes([0.03, 0.95, .05, .05], frameon=0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(.0, .0, 'A', fontsize=26, fontweight='bold')
    ax = fig.add_axes([0.52, 0.95, .05, .05], frameon=0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(.0, .0, 'B', fontsize=26, fontweight='bold')
    ax = fig.add_axes([0.05, 0.4, .05, .05], frameon=0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(.0, .0, 'C', fontsize=26, fontweight='bold')
    fig.savefig(plot_path+'fig1.eps')
    pyplot.show()

if __name__=='__main__':
    generate_data_figure1(data_path='')
    plot_figure1(data_path='',plot_path='')