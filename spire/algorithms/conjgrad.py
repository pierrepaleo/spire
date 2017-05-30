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
from spire.operators.image import grad_tv_smoothed, tv_smoothed, norm2sq, dot

def conjugate_gradient_tv(data, K, Kadj, Lambda, n_it, mu=1e-4, return_all=True, x0=None, recalculate_gradient=50):
    '''
    Conjugate Gradient algorithm to minimize the objective function
    1/2 ||K x - d||_2^2 + Lambda TV_mu (x)
    where TV_mu is the Moreau-Yosida regularization of the Total Variation.

    Parameters
    ------------
    K : forward operator
    Kadj : backward operator, adjoint of K
    data: acquired data
    Lambda : parameter weighting the TV regularization
    mu : parameter of Moreau-Yosida approximation of TV (small positive value)
    n_it : number of iterations
    x0: initial solution estimate
    recalculate_gradient: the gradient is periodically re-computed to avoid error accumulation
    '''

    if x0 is not None:
        x = x0
    else:
        x = 0*Kadj(data) # start from 0
    grad_f = -Kadj(data)
    grad_F = grad_f + Lambda*grad_tv_smoothed(x, mu)
    d = -np.copy(grad_F)

    if return_all: en = np.zeros(n_it)

    for k in range(0, n_it):
        grad_f_old = grad_f
        grad_F_old = grad_F
        ATAd = Kadj(K(d))
        # Calculate step size
        alpha = dot(d, -grad_F_old)/dot(d, ATAd)
        # Update variables
        x = x + alpha*d
        if (k % recalculate_gradient) == 0:
            grad_f = Kadj(K(x)-data)
        else:
            grad_f = grad_f_old + alpha*ATAd
        grad_F = grad_f + Lambda*grad_tv_smoothed(x,mu)
        beta = dot(grad_F, grad_F - grad_F_old)/norm2sq(grad_F_old) # Polak-Ribiere
        if beta < 0:
            beta = 0
        d = -grad_F + beta*d
        # Energy
        if return_all:
            fid = norm2sq(K(x)-data)
            tv = tv_smoothed(x,mu)
            eng = fid+Lambda*tv
            en[k] = eng
            if (k % 10 == 0): # TODO: more flexible
                print("%d : Energy = %e \t Fid = %e\t TV = %e" %(k, eng, fid, tv))

        # Stoping criterion
        if np.abs(alpha) < 1e-15: # TODO : try other bounds
            print("Warning : minimum step reached, interrupting at iteration %d" %k)
            break;
    if return_all: return en, x
    else: return x







def conjugate_gradient(K, Kadj, data, n_it, return_all=True):
    '''
    Conjugate Gradient algorithm for least squares fitting :
    0.5 ||K x - d||_2^2

    Parameters
    -----------
    K : forward operator
    Kadj : backward operator, adjoint of K
    data: acquired data
    n_it : number of iterations
    '''

    x = 0*Kadj(data) # start from 0
    grad_f = -Kadj(data)
    d = -np.copy(grad_f)

    if return_all: en = np.zeros(n_it)

    for k in range(0, n_it):
        grad_f_old = grad_f
        ATAd = Kadj(K(d))
        # Calculate step size
        alpha = dot(d, -grad_f_old)/dot(d, ATAd)
        # Update variables
        x = x + alpha*d
        grad_f = grad_f_old + alpha*ATAd # TODO: re-compute gradient every K iterations to avoid error accumulation
        beta = dot(grad_f, grad_f - grad_f_old)/norm2sq(grad_f_old) # Polak-Ribiere
        if beta < 0:
            beta = 0
        d = -grad_f + beta*d
        # Energy
        if return_all:
            eng = norm2sq(K(x)-data)
            en[k] = eng
            if (k % 10 == 0): # TODO: more flexible
                print("%d : Energy = %e" %(k, eng))

        # Stoping criterion
        if np.abs(alpha) < 1e-15: # TODO : try other bounds
            print("Warning : minimum step reached, interrupting at iteration %d" %k)
            break;
    if return_all: return en, x
    else: return x









