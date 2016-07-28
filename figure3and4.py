"""
Code for generating Figure 3 and 4 of <>. (Data from Figure 1 is required!)

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
import bethe_approximation, synthesis, transforms, __init__, mean_field
from matplotlib import pyplot
import __init__ as ssll


def generate_data_figure3and4(data_path = '../Data/', num_of_iterations=10):
    R, T, N, O = 200, 500, 15, 2
    f = h5py.File(data_path + 'figure1data.h5', 'r')
    theta = f['data']['theta1'].value
    f.close()

    transforms.initialise(N, O)
    psi_true = numpy.empty(T)
    for i in range(T):
        psi_true[i] = transforms.compute_psi(theta[i])
    p = numpy.zeros((T, 2 ** N))
    for i in range(T):
        p[i, :] = transforms.compute_p(theta[i, :])
    fitting_methods = ['exact', 'bethe_hybrid', 'mf']

    f = h5py.File(data_path + 'figure2and3data.h5', 'w')
    f.create_dataset('psi_true', data=psi_true)
    f.create_dataset('theta_true', data=theta)
    for fit in fitting_methods:
        g = f.create_group(fit)
        g.create_dataset('MISE_theta', shape=[num_of_iterations])
        g.create_dataset('MISE_psi', shape=[num_of_iterations])
        g.create_dataset('psi', shape=[num_of_iterations, T])
    f.close()

    for iteration in range(num_of_iterations):
        print 'Iteration %d' % iteration
        spikes = synthesis.generate_spikes(p, R, seed=None)

        for fit in fitting_methods:
            if fit == 'exact':
                emd = __init__.run(spikes, O, map_function='cg',
                                   param_est='exact', param_est_eta='exact')
            else:
                emd = __init__.run(spikes, O, map_function='cg',
                                   param_est='pseudo', param_est_eta=fit)

            psi = numpy.empty(T)

            if fit == 'exact':
                for i in range(T):
                    psi[i] = transforms.compute_psi(emd.theta_s[i])
            elif fit == 'bethe_hybrid':
                for i in range(T):
                    psi[i] = bethe_approximation.compute_eta_hybrid(
                        emd.theta_s[i], N, return_psi=1)[1]
            elif fit == 'mf':
                for i in range(T):
                    eta_mf = mean_field.forward_problem(emd.theta_s[i], N,
                                                        'TAP')
                    psi[i] = mean_field.compute_psi(emd.theta_s[i], eta_mf, N)

            mise_theta = numpy.mean((theta - emd.theta_s) ** 2)
            mise_psi = numpy.mean((psi_true - psi) ** 2)
            f = h5py.File(data_path + 'figure2and3data.h5', 'r+')
            g = f[fit]
            g['MISE_theta'][iteration] = mise_theta
            g['MISE_psi'][iteration] = mise_psi
            if iteration == 0:
                g.create_dataset('theta', data=emd.theta_s)
                g.create_dataset('sigma', data=emd.sigma_s)
            g['psi'][iteration] = psi
            f.close()
            print 'Fitted with %s' % fit

def plot_figure3(data_path='../Data/', plot_path='../Plots/'):
    import h5py
    exact_color = numpy.array([0., 153., 76.]) / 256.
    bethe_color = numpy.array([255., 154., 51.]) / 256.
    mf_color = numpy.array([153., 153., 255.]) / 256.
    D = ssll.transforms.compute_D(15, 2)
    f = h5py.File(data_path+'figure2and3data.h5', 'r')
    theta_true = f['theta_true'].value
    theta_bethe = f['bethe_hybrid']['theta'].value
    theta_mf = f['mf']['theta'].value
    fig = pyplot.figure(figsize=(10, 6))
    T = [50, 150, 300]
    for i, t in enumerate(T):
        ax1 = fig.add_subplot(2, 4, i + 1, aspect='equal')
        ax1.set_frame_on(False)
        ax1.plot([-3., 1.2], [-3., 1.2], color=[.5, .5, .5], linewidth=2)
        ax1.scatter(theta_true[t], theta_bethe[t], c=bethe_color, zorder=4)
        ax1.set_xlim([-3., 1.2])
        ax1.set_ylim([-3., 1.2])
        ymin, ymax = ax1.get_yaxis().get_view_interval()
        xmin, xmax = ax1.get_xaxis().get_view_interval()
        ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                     linewidth=2))
        ax1.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                     linewidth=3))
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.set_xticks(numpy.arange(-2, 2))
        ax1.set_yticks(numpy.arange(-2, 2))
        ax1.set_title('T = %d' % t)
        ax2 = fig.add_subplot(2, 4, i + 5, aspect='equal')
        ax2.set_frame_on(False)
        ax2.plot([-3., 1.2], [-3., 1.2], color=[.5, .5, .5], linewidth=2)
        ax2.scatter(theta_true[t], theta_mf[t], c=mf_color, zorder=4)
        ax2.set_xlim([-3., 1.2])
        ax2.set_ylim([-3., 1.2])
        ymin, ymax = ax2.get_yaxis().get_view_interval()
        xmin, xmax = ax2.get_xaxis().get_view_interval()
        ax2.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                     linewidth=2))
        ax2.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                     linewidth=3))
        ax2.set_xticks(numpy.arange(-2, 2))
        ax2.set_yticks(numpy.arange(-2, 2))
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        if i == 0:
            ax1.set_ylabel('$\hat\\theta_{bethe}$', fontsize=16)
            ax2.set_ylabel('$\hat\\theta_{TAP}$', fontsize=16)
        if i == 1:
            ax2.set_xlabel('$\\theta_{true}$', fontsize=16)

    mse = numpy.empty([3])
    mse[0] = numpy.mean(numpy.sqrt(f['exact']['MISE_theta'].value * D))
    mse[1] = numpy.mean(numpy.sqrt(f['bethe_hybrid']['MISE_theta'].value * D))
    mse[2] = numpy.mean(numpy.sqrt(f['mf']['MISE_theta'].value * D))
    mse_std = numpy.empty(3)
    mse_std[0] = numpy.std(numpy.sqrt(f['exact']['MISE_theta'].value * D))
    mse_std[1] = numpy.std(
        numpy.sqrt(f['bethe_hybrid']['MISE_theta'].value * D))
    mse_std[2] = numpy.std(numpy.sqrt(f['mf']['MISE_theta'].value * D))
    ax = fig.add_axes([.8, .15, .15, .7])
    ax.set_frame_on(False)
    ax.bar([0, 1, 2], mse, width=.5, yerr=mse_std,
           color=[exact_color, bethe_color, mf_color], linewidth=1,
           error_kw=dict(ecolor='k', lw=1, capsize=5, capthick=1))
    ax.set_xlim([-.2, 2.7])
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([.25, 1.25, 2.25])
    ax.set_xticklabels(['exact', 'Bethe', 'TAP'])
    ax.set_yticks([.5, 1.])
    ax.set_ylabel('RMSE $\\theta$', fontsize=16)
    ax = fig.add_axes([0.05, 0.9, .05, .05], frameon=0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(.0, .0, 'A', fontsize=16, fontweight='bold')
    ax = fig.add_axes([0.73, 0.9, .05, .05], frameon=0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(.0, .0, 'B', fontsize=16, fontweight='bold')
    f.close()
    fig.savefig(plot_path+'fig3.eps')
    pyplot.show()


def plot_figure4(data_path='../Data/', plot_path='../Plots/'):
    T = 500
    exact_color = numpy.array([0., 153., 76.]) / 256.
    bethe_color = numpy.array([255., 154., 51.]) / 256.
    mf_color = numpy.array([153., 153., 255.]) / 256.
    f = h5py.File(data_path + 'figure2and3data.h5', 'r')
    psi_true = f['psi_true'].value
    psi_exact = f['exact']['psi'][0, :]
    psi_mf = f['mf']['psi'][0, :]
    psi_bethe = f['bethe_hybrid']['psi'][0, :]
    fig = pyplot.figure(figsize=(10, 3))
    ax1 = fig.add_axes([.1, .155, .6, .8])
    ax1.set_frame_on(False)
    ax1.plot(range(T), numpy.exp(-psi_true), linewidth=3, color='k',
             linestyle='--', zorder=4)
    ax1.plot(range(T), numpy.exp(-psi_exact), linewidth=2, color=exact_color)
    ax1.plot(range(T), numpy.exp(-psi_bethe), linewidth=2, color=bethe_color)
    ax1.plot(range(T), numpy.exp(-psi_mf), linewidth=2, color=mf_color)
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                 linewidth=2))
    ax1.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                 linewidth=3))
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks([0, .05, .1])
    ax1.set_xticks([50, 150, 300])
    ax1.set_ylabel('$p_{\\mathrm{silence}}$', fontsize=16)
    ax1.set_xlabel('Time [AU]', fontsize=16)
    ax1.legend(['True', 'exact', 'bethe', 'TAP'], frameon=0, loc=0)
    mse = numpy.empty([3])
    mse[0] = numpy.mean(numpy.sqrt(f['exact']['MISE_psi'].value))
    mse[1] = numpy.mean(numpy.sqrt(f['bethe_hybrid']['MISE_psi'].value))
    mse[2] = numpy.mean(numpy.sqrt(f['mf']['MISE_psi'].value))
    mse_std = numpy.empty([3])
    mse_std[0] = numpy.std(numpy.sqrt(f['exact']['MISE_psi'].value))
    mse_std[1] = numpy.std(numpy.sqrt(f['bethe_hybrid']['MISE_psi'].value))
    mse_std[2] = numpy.std(numpy.sqrt(f['mf']['MISE_psi'].value))
    ax2 = fig.add_axes([.8, 0.155, .15, .8])
    ax2.set_frame_on(False)
    ax2.bar([0, 1, 2], mse, width=.5, yerr=mse_std,
            color=[exact_color, bethe_color, mf_color],
            error_kw=dict(ecolor='k', lw=1, capsize=5, capthick=1))
    ax2.set_xticklabels(['exact', 'Bethe', 'TAP'])
    ax2.set_xlim([-.2, 2.7])
    ymin, ymax = ax2.get_yaxis().get_view_interval()
    xmin, xmax = ax2.get_xaxis().get_view_interval()
    ax2.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black',
                                 linewidth=2))
    ax2.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black',
                                 linewidth=3))
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_yticks([.05, .1, .15])
    ax2.set_ylabel('RMSE $\psi$', fontsize=16)
    ax2.set_xticks([.25, 1.25, 2.25])
    ax = fig.add_axes([0.05, 0.9, .05, .05], frameon=0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(.0, .0, 'A', fontsize=16, fontweight='bold')
    ax = fig.add_axes([0.73, 0.9, .05, .05], frameon=0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(.0, .0, 'B', fontsize=16, fontweight='bold')
    fig.savefig(plot_path+'fig4.eps')
    fig.show()

if __name__=='__main__':
    generate_data_figure3and4(data_path='', num_of_iterations=2)
    plot_figure3(data_path='', plot_path='')
    plot_figure4(data_path='', plot_path='')
