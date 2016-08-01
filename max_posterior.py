"""
Functions for computing maximum a-posterior probability estimates of natural
parameters given the observed data.

To the original code new gradient descent algorithms are added as conjugate
gradient and BFGS.
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

import probability
import transforms
import pseudo_likelihood



# Named function pointers to MAP estimators
# SEE BOTTOM OF FILE

# Parameters for gradient-ascent methods of MAP estimation
MAX_GA_ITERATIONS = 500
GA_CONVERGENCE = 1e-4

def run(emd, t):
    """
    Computes the MAP estimate of the natural parameters at some timestep, given
    the observed spike patterns at that timestep and the one-step-prediction
    mean and covariance for the same timestep. This function pass the variables
    at time t to the user-specified gradient ascent alogirhtm.
    """
    # Set time bin in pseudo_likelihood
    pseudo_likelihood.time_bin = t
    # Extract observed patterns and one-step predictions for time t
    y_t = emd.y[t,:]
    # Data at time t
    X_t = emd.spikes[t,:,:]
    # Number of runs
    R = emd.R
    # Initial values of natural parameters
    theta_0 = emd.theta_s[t,:]
    # Mean and covariance of one-step prediction density
    theta_o = emd.theta_o[t,:]
    sigma_o = emd.sigma_o[t]
    sigma_o_i = emd.sigma_o_inv[t]
    # Run the user-specified gradient ascent algorithm
    theta_f, sigma_f = emd.max_posterior(y_t, X_t, R, theta_0, theta_o,
                                          sigma_o, sigma_o_i, emd.param_est_eta)

    return theta_f, sigma_f


def newton_raphson(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i, *args):
    """
    TODO update comments to elaborate on how this method differs from the others

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param int t:
        Timestep for which to compute the maximum posterior probability.

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.
    """
    # Initialise the loop guards
    max_dlpo = numpy.inf
    iterations = 0
    # Initialise theta_max to the smooth theta value of the previous iteration
    theta_max = theta_0
    # Iterate the gradient ascent algorithm until convergence or failure
    while max_dlpo > GA_CONVERGENCE:
        # Compute the eta of the current theta values
        p = transforms.compute_p(theta_max)
        eta = transforms.compute_eta(p)
        # Compute the first derivative of the posterior prob. w.r.t. theta_max
        dllk = R * (y_t - eta)
        dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
        dlpo = dllk + dlpr
        # Compute the second derivative of the posterior prob. w.r.t. theta_max
        ddlpo = -R * transforms.compute_fisher_info(p, eta) - sigma_o_i
        # Dot the results to climb the gradient, and accumulate the
        # Small regularization added to avoid singular matrices
        ddlpo_i = numpy.linalg.inv(ddlpo + numpy.finfo(float).eps*\
                                   numpy.identity(eta.shape[0]))
        # Update Theta
        theta_max -= numpy.dot(ddlpo_i, dlpo)
        # Update the look guard
        max_dlpo = numpy.amax(numpy.absolute(dlpo)) / R
        # Count iterations
        iterations += 1
        # Check for check for overrun
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The maximum-a-posterior gradient-ascent '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    return theta_max, -ddlpo_i


def conjugate_gradient(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i, *args):
    """ Fits with `Nonlinear Conjugate Gradient Method
    <https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method>`_.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param int t:
        Timestep for which to compute the maximum posterior probability.

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.

    @author: Christian Donner
    """

    # Initialize theta with previous smoothed theta
    theta_max = theta_0
    # Get p and eta values for current theta
    p = transforms.compute_p(theta_max)
    eta = transforms.compute_eta(p)
    # Compute derivative of posterior
    dllk = R*(y_t - eta)
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    dlpo = dllk + dlpr
    # Initialize stopping criterion variables
    max_dlpo = 1.
    iterations = 0
    # Get theta gradient
    d_th = dlpo
    # Set initial search direction
    s = dlpo
    # Compute line search
    theta_max, dlpo, p, eta = line_search(theta_max, y_t, R, p, s, dlpo,
                                          theta_o, sigma_o_i)

    # Iterate until convergence or failure
    while max_dlpo > GA_CONVERGENCE:

        # Set current theta gradient to previous
        d_th_prev = d_th
        # The new theta gradient
        d_th = dlpo
        # Calculate beta
        beta = compute_beta(d_th, d_th_prev)
        # New search direction
        s = d_th + beta * s
        # Line search
        theta_max, dlpo, p, eta = line_search(theta_max, y_t, R, p, s, dlpo,
                                              theta_o, sigma_o_i)
        # Get maximal entry of log posterior grad divided by number of trials
        max_dlpo = numpy.amax(numpy.absolute(dlpo)) / R
        # Count iterations
        iterations += 1
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The maximum-a-posterior conjugate-gradient '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    # Compute final covariance matrix
    ddllk = - R*transforms.compute_fisher_info(p, eta)
    ddlpo = ddllk - sigma_o_i
    ddlpo_i = numpy.linalg.inv(ddlpo)

    return theta_max, -ddlpo_i


def bfgs(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i, *args):
    """ Fits due to `Broyden-Fletcher-Goldfarb-Shanno algorithm
    <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%
    80%93Shanno_algorithm>`_.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param int t:
        Timestep for which to compute the maximum posterior probability.

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.

    @author: Christian Donner
    """

    # # Initialize theta with previous smoothed theta
    theta_max = theta_0
    # Get p and eta values for current theta
    p = transforms.compute_p(theta_max)
    eta = transforms.compute_eta(p)
    # Initialize the estimate of the inverse fisher info
    ddlpo_i_e = numpy.identity(theta_max.shape[0])
    # Compute derivative of posterior
    dllk = R*(y_t - eta)
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    dlpo = dllk + dlpr
    # Initialize stopping criterion variables
    max_dlpo = 1.
    iterations = 0

    # Iterate until convergence or failure
    while max_dlpo > GA_CONVERGENCE:
        # Compute direction for line search
        s_dir = numpy.dot(dlpo, ddlpo_i_e)
        # Set theta to old theta
        theta_prev = numpy.copy(theta_max)
        # Set current log posterior gradient to previous
        dlpo_prev = dlpo
        # Perform line search
        theta_max, dlpo, p, eta = line_search(theta_max, y_t, R, p, s_dir, dlpo,
                                              theta_o, sigma_o_i)
        # Get the difference between old and new theta
        d_theta = theta_max - theta_prev
        # Difference in log posterior gradients
        dlpo_diff = dlpo_prev - dlpo
        # Project gradient change on theta change
        dlpo_diff_dth = numpy.inner(dlpo_diff, d_theta)
        # Compute estimate of covariance matrix with Sherman-Morrison Formula
        a = (dlpo_diff_dth + \
             numpy.dot(dlpo_diff, numpy.dot(ddlpo_i_e, dlpo_diff.T)))*\
            numpy.outer(d_theta, d_theta)
        b = numpy.inner(d_theta, dlpo_diff)**2
        c = numpy.dot(ddlpo_i_e, numpy.outer(dlpo_diff, d_theta)) + \
            numpy.outer(d_theta, numpy.inner(dlpo_diff, ddlpo_i_e))
        d = dlpo_diff_dth
        ddlpo_i_e += (a/b - c/d)
        # Get maximal entry of log posterior grad divided by number of trials
        max_dlpo = numpy.amax(numpy.absolute(dlpo)) / R
        # Count iterations
        iterations += 1
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The maximum-a-posterior bfgs-gradient '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    # Compute final covariance matrix
    ddllk = -R*transforms.compute_fisher_info(p, eta)
    ddlpo = ddllk - sigma_o_i
    ddlpo_i = numpy.linalg.inv(ddlpo)

    return theta_max, -ddlpo_i


def line_search(theta_max, y, R, p, s, dlpo, theta_o, sigma_o_i):
    """ Searches the minimum on a line with quadratic approximation

    :param numpy.ndarray theta_max:
        Starting point on the line
    :param numpy.ndarray y:
        Empirical mean of the data (sufficient statistics)
    :param int R:
        Number of trials
    :param numpy.ndarray p:
        Probability for each pattern
    :param numpy.ndarray s:
        Direction that is searched in
    :param numpy.ndarray dlpo:
        Derivative of of th posterior at the current theta
    :param numpy.ndarray theta_o:
        One-step prediction of theta
    :param numpy.ndarray sigma_o_i:
        One-step prediction of the covariance matrix

    :returns
        Tuple containing the minimum on the line, the log posterior gradient,
        the current p and current eta vector

    This method approximates at each point the log posterior quadratically
    and searches iteratively for the minimum.

    @author: Christian Donner
    """
    y_s = numpy.dot(y, s)
    # Project theta on p_map
    theta_p = transforms.p_map.dot(theta_max)
    # Project p-map on search direction
    p_map_s = transforms.p_map.dot(s)
    # Projected eta on search direction
    eta_s = numpy.dot(p_map_s, p)
    # Project inverse one-step covariance matrix on search direction
    sigma_o_i_s = numpy.dot(sigma_o_i, s)
    # Project gradient of log posterior on search direction
    dlpo_s = numpy.dot(dlpo, s)
    # Get Metric of fisher info along s direction
    s_G_s = R*(numpy.dot(p_map_s, p*p_map_s) - eta_s**2) + \
            numpy.dot(s, sigma_o_i_s)
    # Initialize iteration variable and alpha
    dalpha = numpy.inf
    alpha = 0
    snorm = numpy.sum(numpy.absolute(s))
    while dalpha*snorm > 1e-2:
        # Compute alpha due to gradient
        alpha_new = alpha + dlpo_s/s_G_s
        # If new alpha is negative take the half of old alpha
        if alpha_new < 0:
            alpha /= 2.
            dalpha = alpha
        # Else take new
        else:
            dalpha = numpy.absolute(alpha - alpha_new)
            alpha = alpha_new
        # Update theta
        theta_tmp = theta_max + alpha*s
        # Compute new psi
        psi_new = numpy.log(numpy.sum(numpy.exp(theta_p + alpha*p_map_s)))
        # psi = numpy.log(p*numpy.exp(alpha*p_map_s))
        p = numpy.exp(theta_p + alpha*p_map_s - psi_new)
        # Project eta on search direction
        eta_s = numpy.dot(p_map_s, p)
        # Project fisher information on search direction
        s_G_s = R*(numpy.dot(p_map_s, p*p_map_s) - eta_s**2) + \
                numpy.dot(s, sigma_o_i_s)
        # Compute log posterior gradient projected on s
        dllk_s = R*(y_s - eta_s)
        dlpr_s = -numpy.dot(sigma_o_i_s, theta_tmp - theta_o)
        dlpo_s = dllk_s + dlpr_s
    # return optimized theta and current gradient of log posterior

    eta = transforms.compute_eta(p)
    dllk = R*(y - eta)
    dlpr = -numpy.dot(sigma_o_i, theta_tmp - theta_o)
    dlpo = dllk + dlpr
    return theta_tmp, dlpo, p, eta


def compute_beta(df, dfp, s=None, which='PR'):
    """ Computes the beta Polak Ribiere Formula

    :param numpy.ndarray df:
        gradient of function to minimize
    :param numpy.ndarray dfp:
        previous gradient of function to minimize

    :returns float:
        result of Polak Ribiere Formula

    @author: Christian Donner
    """

    # Polak Ribiere Formula
    if which == 'PR':
        beta = float(numpy.dot(df, (df - dfp)) / numpy.dot(dfp, dfp))
    elif which == 'HS':
        if numpy.allclose(df, dfp):
            return 0
        beta = -float(numpy.dot(df, (df - dfp)) / numpy.dot(s, (df - dfp)))
    return numpy.amax([0, beta])


# Named function pointers to MAP estimators
functions = {'nr': newton_raphson,
             'cg': conjugate_gradient,
             'bf': bfgs}