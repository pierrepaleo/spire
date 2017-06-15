#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016, European Synchrotron Radiation Facility
# Main author: Pierre Paleo <pierre.paleo@esrf.fr>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of SPIRE nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from __future__ import division
import numpy as np
from spire.operators.misc import power_method
from spire.operators.image import gradient, div, norm1, norm2sq, proj_l2, proj_linf
from math import sqrt


def chambolle_pock_tv(data, K, Kadj, Lambda, L=None,  n_it=100, return_all=True, x0=None, pos_constraint=False):
    '''
    Chambolle-Pock algorithm for Total Variation regularization.
    The following objective function is minimized : ||K x - d||_2^2 + Lambda TV(x)

    Parameters
    -----------

    K : function
        forward operator
    Kadj : function
        backward operator
    Lambda : float
        weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : float
        norm of the operator [P, Lambda*grad] (see power_method)
    n_it : int
        number of iterations
    return_all: bool
        if True, an array containing the values of the objective function will be returned
    x0: numpy.ndarray
        initial solution estimate
    '''

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)

    sigma = 1.0/L
    tau = 1.0/L

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = gradient(x)
    q = 0*data
    x_tilde = 1.0*x
    theta = 1.0

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        # For isotropic TV, the prox is a projection onto the L2 unit ball.
        # For anisotropic TV, this is a projection onto the L-infinity unit ball.
        p = proj_linf(p + sigma*gradient(x_tilde), Lambda)
        q = (q + sigma*K(x_tilde) - sigma*data)/(1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x + tau*div(p) - tau*Kadj(q)
        if pos_constraint:
            x[x<0] = 0
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x



def chambolle_pock_l1_tv(data, K, Kadj, Lambda, L=None,  n_it=100, return_all=True, x0=None):
    '''
    Chambolle-Pock algorithm for L1-TV.
    The following objective function is minimized : ||K x - d||_1 + Lambda TV(x).
    This method is recommended against L2-TV for noise with strong outliers (eg. salt & pepper).

    Parameters
    ------------
    K : forward operator
    Kadj : backward operator
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] (see power_method)
    n_it : number of iterations
    return_all: if True, an array containing the values of the objective function will be returned
    x0: initial solution estimate
    '''

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)
    sigma = 1.0/L
    tau = 1.0/L

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        p = proj_l2(p + sigma*gradient(x_tilde), Lambda)
        q = proj_linf(q + sigma*K(x_tilde) - sigma*data) # Here the projection onto the l-infinity ball is absolutely required !

        # Update primal variables
        x_old = x
        x = x + tau*div(p) - tau*Kadj(q)
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x






def chambolle_pock_kl_tv(data, K, Kadj, Lambda, L=None,  n_it=100, return_all=True, x0=None):
    '''
    Chambolle-Pock algorithm for KL-TV.
    The following objective function is minimized : KL(K x , d) + Lambda TV(x)
    Where KL(x, y) is a modified Kullback-Leibler divergence.
    This method might be more effective than L2-TV for Poisson noise.

    K : forward operator
    Kadj : backward operator
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] (see power_method)
    n_it : number of iterations
    return_all: if True, an array containing the values of the objective function will be returned
    x0: initial solution estimate
    '''

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)
    sigma = 1.0/L
    tau = 1.0/L

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0

    #
    O = np.ones_like(q)
    #

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        tmp = q + sigma*K(x_tilde)
        q = 0.5*(O + tmp - np.sqrt((tmp - O)**2 + 4*sigma*data))
        tmp = p + sigma*gradient(x_tilde)
        p = Lambda*(tmp)/np.maximum(Lambda, np.abs(tmp))

        # Update primal variables
        x_old = x
        x = x + tau*div(p) - tau*Kadj(q)
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x




# ------------------------------------------------------------------------------
# ------------- Various version of Chambolle-Pock algorithm---------------------
# ------------------------------------------------------------------------------



# Accelerated algorithm for F *or* G^* uniformly convex (Algorithm 2, p. 15)
def chambolle_pock_tv2(data, K, Kadj, Lambda, L=None, gamma=1.0, theta=1.0, n_it=100, return_all=True, x0=None):

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)

    sigma = 1.0/L
    tau = 1.0/L
    #gamma = 0.5 # Should be the uniform convexity parameter of "F"

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    # theta = 1.0 # theta = 0 gives another fast algorithm

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        p = proj_linf(p + sigma*gradient(x_tilde), Lambda)
        q = (q + sigma*K(x_tilde) - sigma*data)/(1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x + tau*div(p) - tau*Kadj(q)
        theta = 1./sqrt(1.+2*gamma*tau)
        tau = theta*tau
        sigma = sigma/theta
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x






def chambolle_pock_tv_relaxed(data, K, Kadj, Lambda, L=None,  n_it=100, return_all=True, x0=None):

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)

    sigma = 1.0/L
    tau = 1.0/L

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update primal variables
        x_old = x
        x = x + tau*div(p) - tau*Kadj(q)
        # Update dual variables
        p = proj_linf(p + sigma*gradient(x + theta*(x - x_old)), Lambda)
        q = (q + sigma*K(x + theta*(x - x_old)) - sigma*data)/(1.0 + sigma)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x



def chambolle_pock_tv_relaxed2(data, K, Kadj, Lambda, L=None, rho=1.9, tau = None, n_it=100, return_all=True, x0=None):

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)

    if tau is None: tau = 1.0/L
    sigma = 1.0/(tau*L*L)

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update primal variables
        x_tilde = x + tau*div(p) - tau*Kadj(q)
        # Update dual variables
        p_tilde = proj_linf(p + sigma*gradient(2*x_tilde-x), Lambda)
        q_tilde = (q + sigma*K(2*x_tilde-x) - sigma*data)/(1.0 + sigma)
        # Relaxed version
        #~ x = rho*x_tilde + (1-rho)*x
        #~ p = rho*p_tilde + (1-rho)*p
        #~ q = rho*q_tilde + (1-rho)*q
        x = x_tilde + (rho - 1)*(x_tilde - x)
        p = p_tilde + (rho - 1)*(p_tilde - p)
        q = q_tilde + (rho - 1)*(q_tilde - q)

        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x







# Preconditioned version described in [4]
# [4] T. Pock, A. Chambolle, "Diagonal preconditioning for first order primal-dual algorithms in convex optimization",
#   2011, International Conference on Computer Vision
#
#~ @INPROCEEDINGS{6126441,
#~ author={T. Pock and A. Chambolle},
#~ booktitle={2011 International Conference on Computer Vision},
#~ title={Diagonal preconditioning for first order primal-dual algorithms in convex optimization},
#~ year={2011},
#~ pages={1762-1769},
#~ doi={10.1109/ICCV.2011.6126441},
#~ ISSN={1550-5499},
#~ month={Nov},}

def chambolle_pock_tv_precond(data, K, Kadj, Lambda, n_it=100, return_all=True, x0=None, pos_constraint=False):

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = x
    theta = 1.0

    # Compute the diagonal preconditioner "Sigma" for alpha=1
    # Assuming K is a positive integral operator
    Sigma_k = 1./K(np.ones_like(x))
    Sigma_grad = 1/2.0
    Sigma = 1/(1./Sigma_k + 1./Sigma_grad)
    # Compute the diagonal preconditioner "Tau" for alpha = 1
    # Assuming Kadj is a positive operator
    Tau = 1./(Kadj(np.ones_like(data)) + 2.)

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update primal variables
        x_old = x
        x = x + Tau*div(p) - Tau*Kadj(q)
        if pos_constraint:
            x[x<0] = 0
        # Update dual variables
        p = proj_linf(p + Sigma_grad*gradient(x + theta*(x - x_old)), Lambda) # For discrete gradient, sum|D_i,j| = 2 along lines or cols
        q = (q + Sigma_k*K(x + theta*(x - x_old)) - Sigma_k*data)/(1.0 + Sigma_k) # <=
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x


def chambolle_pock_tv_l1_precond(data, K, Kadj, Lambda, n_it=100, return_all=True, x0=None):

    x = 0*Kadj(data) if x0 is None else x0
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0

    # Compute the diagonal preconditioner "Sigma" for alpha=1
    # Assuming K is a positive integral operator
    Sigma_k = 1./K(np.ones_like(x))
    Sigma_grad = 1/2.0
    Sigma = 1/(1./Sigma_k + 1./Sigma_grad)
    # Compute the diagonal preconditioner "Tau" for alpha = 1
    # Assuming Kadj is a positive operator
    Tau = 1./(Kadj(np.ones_like(data)) + 2.)

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update primal variables
        x_old = x
        x = x + Tau*div(p) - Tau*Kadj(q)
        # Update dual variables
        p = proj_linf(p + Sigma_grad*gradient(x + theta*(x - x_old)), Lambda) # For discrete gradient, sum|D_i,j| = 2 along lines or cols
        #q = (q + Sigma_k*K(x + theta*(x - x_old)) - Sigma_k*data)/(1.0 + Sigma_k) # <=
        q = proj_linf(q + Sigma_k*K(x + theta*(x - x_old)) - Sigma_k*data)
        # Calculate norms
        if return_all:
            fidelity = norm1(K(x)-data)
            tv = norm1(gradient(x))
            energy = fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x







#
# Wavelets + TV
#

def chambolle_pock_tv_wavelets(data, K, Kadj, W, Lambda1, Lambda2, L=None, n_it=100, return_all=True, x0=None):


    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)

    sigma = 1.0/L
    tau = 1.0/L

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        p = proj_linf(p + sigma*gradient(x_tilde), Lambda1)
        q = (q + sigma*K(x_tilde) - sigma*data)/(1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x + tau*div(p) - tau*Kadj(q)
        #
        W.set_image(x)
        W.forward()
        W.soft_threshold(Lambda2, 0, 1)
        wnorm1 = W.norm1()
        W.inverse()
        x = W.image
        #
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda1*tv + Lambda2*wnorm1
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x













def chambolle_pock_tv_l2(data, K, Kadj, U, Uadj, Lambda, Lambda2, L=None,  n_it=100, return_all=True, x0=None):

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        Lr = power_method(U, Uadj, data, 20)
        L = sqrt(8. + L**2 + Lr**2) * 1.2
        print("L = %e" % L)

    sigma = 1.0/L
    tau = 1.0/L

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    r = 0*x
    x_tilde = 0*x
    theta = 1.0

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        p = proj_linf(p + sigma*gradient(x_tilde), Lambda)
        q = (q + sigma*K(x_tilde) - sigma*data)/(1.0 + sigma)
        r = (r + sigma*U(x_tilde))/(1.0 + sigma/Lambda2)
        # Update primal variables
        x_old = x
        x = x + tau*div(p) - tau*Kadj(q) - tau*Uadj(r)
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x








def _soft_thresh(x, beta):
    return np.maximum(np.abs(x)-beta, 0)*np.sign(x)


def pinv_fourier(img, kern, tau=1., bw=10):
    n_r, n_c = img.shape
    shp2 = (2*n_r, 2*n_c)
    img_f = np.fft.fft2(expand_reflect(img, bw))
    kern_f = np.fft.fft2(kern, shp2)
    res = np.fft.ifft2(img_f / (1. + tau*kern_f))[n_r//2:n_r//2+n_r, n_c//2:n_c//2+n_c] #.real
    return np.abs(res)



#~ def pinv_fourier(img, kern, tau=1., bw=10):
    #~ n_r, n_c = img.shape
    #~ shp2 = (2*n_r, 2*n_c)
    #~ img_f = np.fft.fft2(img, shp2)
    #~ kern_f = np.fft.fft2(kern, shp2)
    #~ res = np.fft.ifft2(img_f / (1. + tau*kern_f))[:n_r, :n_c]
    #~ return np.abs(res)




def chambolle_pock_deblur_tv(img, G, kern, Lambda, L=None, n_it=100, return_all=True, bw=10):
    """
    prototype
    """

    if L is None: L = 2.83 # sqrt(8)
    sigma = 1.0/L
    tau = 1.0/L
    theta = 1.0

    x = 0*img
    x_tilde = 0*x
    y = 0*x

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # y_{k+1} = prox_{sigma G^*} (y_k + sigma K xtilde_k)
        y = _soft_thresh(y + sigma*gradient(x_tilde), Lambda*sigma)
        # x_{k+1} = prox_{tau F} (x_n - tau K^* y_{k+1})
        x_old = np.copy(x)
        x = pinv_fourier(x + tau*div(y) + tau*G(img), kern, tau=tau, bw=bw)
        # xtilde{k+1} = x_{k+1} + theta (x_{k+1} - x_k)
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(G(x)-img)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x








def chambolle_pock_laplace(data, K, Kadj, Lambda, L=None,  n_it=100, return_all=True):

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8.**2 + L**2) * 1.2 # Laplacian is self-adjoint, and have a norm 8 (squared of gradient)
        print("L = %e" % L)

    sigma = 1.0/L
    tau = 1.0/L

    x = 0*Kadj(data)
    p = 0*laplacian(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        # For isotropic TV, the prox is a projection onto the L2 unit ball.
        # For anisotropic TV, this is a projection onto the L-infinity unit ball.
        p = proj_linf(p + sigma*laplacian(x_tilde), Lambda)
        q = (q + sigma*K(x_tilde) - sigma*data)/(1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x - tau*laplacian(p) - tau*Kadj(q)
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(laplacian(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x














# ******************************************************************************
# C-P for L1-Wavelets. The two following work quite well, although converging
# quite slowly (it is even worse for a "fully split" problem).
# Fortunately, the preconditioned version works well (although still slow), and
# the continuation method should work (as restarting from x0 works in this case)
# ******************************************************************************








def chambolle_pock_l1_wavelets(data, W, K, Kadj, Lambda, L=None,  n_it=100, return_all=True, x0=None, pos_constraint=False):

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)

    sigma = 1.0/L
    tau = 1.0/L

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = gradient(x)
    q = 0*data
    x_tilde = 1.0*x
    theta = 1.0

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        # For isotropic TV, the prox is a projection onto the L2 unit ball.
        # For anisotropic TV, this is a projection onto the L-infinity unit ball.
        q = proj_linf(q + sigma*K(x_tilde) - sigma*data)
        # Update primal variables
        x_old = x
        W.set_image(x - tau*Kadj(q))
        W.forward()
        W.soft_threshold(Lambda*tau, do_threshold_appcoeffs=1)
        W.inverse()
        x = W.image
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x








def chambolle_pock_l1_wavelets_precond(data, W, K, Kadj, Lambda, n_it=100, return_all=True, x0=None, pos_constraint=False):

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = x
    theta = 1.0

    # Compute the diagonal preconditioner "Sigma" for alpha=1
    # Assuming K is a positive integral operator
    Sigma_k = 1./K(np.ones_like(x))
    Sigma_grad = 1/2.0
    Sigma = 1/(1./Sigma_k + 1./Sigma_grad)
    # Compute the diagonal preconditioner "Tau" for alpha = 1
    # Assuming Kadj is a positive operator
    Tau = 1./(Kadj(np.ones_like(data)) + 2.)

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update primal variables
        x_old = x
        W.set_image(x - Tau*Kadj(q))
        W.forward()
        W.soft_threshold(Lambda, do_threshold_appcoeffs=1)
        W.inverse()
        x = W.image
        # Update dual variables
        q = proj_linf(q + Sigma_k*K(x + theta*(x - x_old)) - Sigma_k*data) # <=
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_all: return en, x
    else: return x









