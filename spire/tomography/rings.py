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


# ------------------------------------------------------------------------------
# --------- Rings artifacts correction : Titarenko's algorithm
# Taken from tomopy :
#
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.
# ------------------------------------------------------------------------------

# De-stripe : Titarenko algorithm
def remove_stripe_ti(sino, nblock=0, alpha=None):
    if alpha is None: alpha = np.std(sino)
    if nblock == 0:
        d1 = _ring(sino, 1, 1)
        d2 = _ring(sino, 2, 1)
        p = d1 * d2
        d = np.sqrt(p + alpha * np.abs(p.min()))
    else:
        size = int(sino.shape[0] / nblock)
        d1 = _ringb(sino, 1, 1, size)
        d2 = _ringb(sino, 2, 1, size)
        p = d1 * d2
        d = np.sqrt(p + alpha * np.fabs(p.min()))
    return d


def _kernel(m, n):
    v = [[np.array([1, -1]),
          np.array([-3 / 2, 2, -1 / 2]),
          np.array([-11 / 6, 3, -3 / 2, 1 / 3])],
         [np.array([-1, 2, -1]),
          np.array([2, -5, 4, -1])],
         [np.array([-1, 3, -3, 1])]]
    return v[m - 1][n - 1]


def _ringMatXvec(h, x):
    s = np.convolve(x, np.flipud(h))
    u = s[np.size(h) - 1:np.size(x)]
    y = np.convolve(u, h)
    return y


def _ringCGM(h, alpha, f):
    x0 = np.zeros(np.size(f))
    r = f - (_ringMatXvec(h, x0) + alpha * x0)
    w = -r
    z = _ringMatXvec(h, w) + alpha * w
    a = np.dot(r, w) / np.dot(w, z)
    x = x0 + np.dot(a, w)
    B = 0
    for i in range(1000000):
        r = r - np.dot(a, z)
        if np.linalg.norm(r) < 0.0000001:
            break
        B = np.dot(r, z) / np.dot(w, z)
        w = -r + np.dot(B, w)
        z = _ringMatXvec(h, w) + alpha * w
        a = np.dot(r, w) / np.dot(w, z)
        x = x + np.dot(a, w)
    return x


def _get_parameter(x):
    return 1 / (2 * (x.sum(0).max() - x.sum(0).min()))


def _ring(sino, m, n):
    mysino = np.transpose(sino)
    R = np.size(mysino, 0)
    N = np.size(mysino, 1)

    # Remove NaN.
    pos = np.where(np.isnan(mysino) is True)
    mysino[pos] = 0

    # Parameter.
    alpha = _get_parameter(mysino)

    # Mathematical correction.
    pp = mysino.mean(1)
    h = _kernel(m, n)
    f = -_ringMatXvec(h, pp)
    q = _ringCGM(h, alpha, f)

    # Update sinogram.
    q.shape = (R, 1)
    K = np.kron(q, np.ones((1, N)))
    new = np.add(mysino, K)
    newsino = new.astype(np.float32)
    return np.transpose(newsino)


def _ringb(sino, m, n, step):
    mysino = np.transpose(sino)
    R = np.size(mysino, 0)
    N = np.size(mysino, 1)

    # Remove NaN.
    pos = np.where(np.isnan(mysino) is True)
    mysino[pos] = 0

    # Kernel & regularization parameter.
    h = _kernel(m, n)

    # Mathematical correction by blocks.
    nblock = int(N / step)
    new = np.ones((R, N))
    for k in range(0, nblock):
        sino_block = mysino[:, k * step:(k + 1) * step]
        alpha = _get_parameter(sino_block)
        pp = sino_block.mean(1)

        f = -_ringMatXvec(h, pp)
        q = _ringCGM(h, alpha, f)

        # Update sinogram.
        q.shape = (R, 1)
        K = np.kron(q, np.ones((1, step)))
        new[:, k * step:(k + 1) * step] = np.add(sino_block, K)
    newsino = new.astype(np.float32)
    return np.transpose(newsino)



# ------------------------------------------------------------------------------
# --------- Rings artifacts correction : Münch Et Al. algorithm
# Copyright (c) 2013, Elettra - Sincrotrone Trieste S.C.p.A.
# All rights reserved.
# ------------------------------------------------------------------------------


# De-stripe : Munch Et Al algorithm
def munchetal_filter(im, wlevel, sigma, wname='db15'):
    # Wavelet decomposition:
    coeffs = pywt.wavedec2(im.astype(np.float32), wname, level=wlevel)
    coeffsFlt = [coeffs[0]]
    # FFT transform of horizontal frequency bands:
    for i in range(1, wlevel + 1):
        # FFT:
        fcV = np.fft.fftshift(np.fft.fft(coeffs[i][1], axis=0))
        my, mx = fcV.shape
        # Damping of vertical stripes:
        damp = 1 - np.exp(-(np.arange(-np.floor(my / 2.), -np.floor(my / 2.) + my) ** 2) / (2 * (sigma ** 2)))
        dampprime = np.kron(np.ones((1, mx)), damp.reshape((damp.shape[0], 1)))
        fcV = fcV * dampprime
        # Inverse FFT:
        fcVflt = np.real(np.fft.ifft(np.fft.ifftshift(fcV), axis=0))
        cVHDtup = (coeffs[i][0], fcVflt, coeffs[i][2])
        coeffsFlt.append(cVHDtup)

    # Get wavelet reconstruction:
    im_f = np.real(pywt.waverec2(coeffsFlt, wname))
    # Return image according to input type:
    if (im.dtype == 'uint16'):
        # Check extrema for uint16 images:
        im_f[im_f < np.iinfo(np.uint16).min] = np.iinfo(np.uint16).min
        im_f[im_f > np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max
        # Return filtered image (an additional row and/or column might be present):
        return im_f[0:im.shape[0], 0:im.shape[1]].astype(np.uint16)
    else:
        return im_f[0:im.shape[0], 0:im.shape[1]]

