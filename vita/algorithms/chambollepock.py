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
# * Neither the name of VITA nor the names of its
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
from vita.operators.misc import power_method
from vita.operators.image import gradient, div, norm1, norm2sq, proj_l2, proj_linf
from math import sqrt


def chambolle_pock_tv(data, K, Kadj, Lambda, L=None,  n_it=100, return_all=True):
    '''
    Chambolle-Pock algorithm for Total Variation regularization.
    The following objective function is minimized :
        ||K*x - d||_2^2 + Lambda*TV(x)

    K : forward operator
    Kadj : backward operator
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] (see power_method)
    n_it : number of iterations
    return_all: if True, an array containing the values of the objective function will be returned
    '''

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)

    sigma = 1.0/L
    tau = 1.0/L

    x = 0*Kadj(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0

    if return_all: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        # For anisotropic TV, the prox is a projection onto the L2 unit ball.
        # For anisotropic TV, this is a projection onto the L-infinity unit ball.
        p = proj_linf(p + sigma*gradient(x_tilde), Lambda)
        q = (q + sigma*K(x_tilde) - sigma*data)/(1.0 + sigma)
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


def chambolle_pock_l1_tv(data, K, Kadj, Lambda, L=None,  n_it=100, return_all=True):
    '''
    Chambolle-Pock algorithm for L1-TV.
    The following objective function is minimized :
        ||K*x - d||_1 + Lambda*TV(x)
    This method is recommended against L2-TV for noise with strong outliers (eg. salt & pepper).

    K : forward operator
    Kadj : backward operator
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] (see power_method)
    n_it : number of iterations
    return_all: if True, an array containing the values of the objective function will be returned
    '''

    if L is None:
        print("Warn: chambolle_pock(): Lipschitz constant not provided, computing it with 20 iterations")
        L = power_method(K, Kadj, data, 20)
        L = sqrt(8. + L**2) * 1.2
        print("L = %e" % L)
    sigma = 1.0/L
    tau = 1.0/L

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







