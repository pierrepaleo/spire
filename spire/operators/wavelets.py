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
from math import sqrt, log
from spire.operators.image import norm2sq, norm1
try:
    import pywt
except ImportError:
    raise ImportError("Please install pywt before using Wavelets wrapper")


def _circular_shift(img, i, j):
    Nr, Nc = img.shape
    sr, sc = i%Nr, j%Nc
    res = np.empty_like(img)
    res[sr:, sc:] = img[:-sr, :-sc]
    res[sr:, :sc] = img[:-sr, -sc:]
    res[:sr, sc:] = img[-sr:, :-sc]
    res[:sr, :sc] = img[-sr:, -sc:]
    return res

def _ilog2(i):
    l = 0
    while (i>>1):
        i = i>>1
        l+=1
    return l


def _soft_thresh(x, beta):
    return np.maximum(np.abs(x)-beta, 0)*np.sign(x)


def _proj_linf_img(q, lambda_=1):
    '''
    Projection onto the L-infinity unit ball :
    '''
    res = np.copy(q)
    res[res > lambda_] = lambda_ * np.sign(res[res > lambda_])
    return res



def soft_threshold_coeffs(w, beta, do_threshold_appcoeffs=False): # in-place !
    if do_threshold_appcoeffs: self.a = _soft_thresh(w.a, beta) # FIXME : do I have to also ST the app coeffs ?
    L = len(w.h)
    for k in range(0, L):
        w.h[k] = _soft_thresh(w.h[k], beta)
        w.v[k] = _soft_thresh(w.v[k], beta)
        w.d[k] = _soft_thresh(w.d[k], beta)




class WaveletCoeffs():
    '''
    Wavelets class.

    Attributes
    ----------
    wname : name of the Wavelet transform ('haar', 'db10', ...)
    levels : number of transform levels. Max is log2(max(image_width, image_heigh))-1
    do_random_shifts : if True, a random circular shift of the image is done before forward transform ;
        and the corresponding opposed circular shift is done before the inverse transform.
        It is a way to achieve translation invariance for Discrete Wavelet Transform when using iterative methods.
    shifts : if do_random_shifts is True, tuple containing the current shifts
    a : approximation coefficients (numpy array)
    h : list of "levels" arrays containing horizontal coefficients at various scales
    v : list of "levels" arrays containing vertical coefficients at various scales
    d : list of "levels" arrays containing details coefficients at various scales
    '''
    def __init__(self, data, wname=None, levels=None, do_random_shifts=False):
        '''
        Initialization of the Wavelet class.

        wname : name of the Wavelet
        levels : number of levels of decomposition. If not provided, the maximum number of levels will be used
        randomshifts : use Random Shifts for the decomposition/reconstruction (improves the quality for iterative reconstruction)
        '''

        # Deep copy of a set of wavelet coefficients
        if isinstance(data, WaveletCoeffs):
            self.levels = data.levels
            self.wname = data.wname
            self.do_random_shifts = data.do_random_shifts
            self.shifts = data.shifts
            self.a = np.copy(data.a)
            self.h = []
            self.v = []
            self.d = []
            for k in range(0, levels):
                self.h.append(np.copy(data.h[k]))
                self.v.append(np.copy(data.v[k]))
                self.d.append(np.copy(data.d[k]))

       # Wavelet decomposition from an image
        else:
            if levels is None: self.levels = _ilog2(max(data.shape))-1
            else: self.levels = int(levels)
            if (data.shape[0] != data.shape[1]) and (do_random_shifts): # TODO
                raise NotImplementedError('Random shifts can only be used with square image for now')
            self.do_random_shifts = bool(do_random_shifts)
            if self.do_random_shifts is True:
                self.shifts = np.random.randint(data.shape[0]-1, size=2)+1
            if wname is None: self.wname = 'haar'
            else: self.wname = wname

            # Compute DWT
            data2 = _circular_shift(data, self.shifts[0], self.shifts[1]) if self.do_random_shifts is True else data
            S = pywt.wavedec2(data2, self.wname, mode='per', level=self.levels)
            # app coeffs -- numpy.ndarray
            self.a = S[0]
            # horizontal, vertical and diagonal  detail coeffs -- lists (instead of pywt tuples)
            self.h = []
            self.v = []
            self.d = []
            for k in range(1, self.levels+1): # store in decreasing order of levels (high index = big image)
                self.h.append(S[k][0])
                self.v.append(S[k][1])
                self.d.append(S[k][2])


    def __add__(self, W):
        res = WaveletCoeffs(W)
        res.a += self.a
        for k in range(0, len(res.h)):
            res.h[k] += self.h[k]
            res.v[k] += self.v[k]
            res.d[k] += self.d[k]
        return res

    def __sub__(self, W):
        res = WaveletCoeffs(self)
        res.a -= W.a
        for k in range(0, len(res.h)):
            res.h[k] -= W.h[k]
            res.v[k] -= W.v[k]
            res.d[k] -= W.d[k]
        return res

    def __rmul__(self, num):
        res = WaveletCoeffs(self)
        res.a *= num
        for k in range(0, len(res.h)):
            res.h[k] *= num
            res.v[k] *= num
            res.d[k] *= num
        return res

    def __mul__(self, W):
        res = WaveletCoeffs(self)
        res.a *= W.a
        for k in range(0, len(res.h)):
            res.h[k] *= W.h[k]
            res.v[k] *= W.v[k]
            res.d[k] *= W.d[k]
        return res


    def norm_fro(self):
        res = 0
        res += norm2sq(self.a)
        for k in range(0, len(self.h)):
            res += norm2sq(self.h[k])
            res += norm2sq(self.v[k])
            res += norm2sq(self.d[k])
        return sqrt(res)

    def norm2sq(self):
        res = 0
        res += norm2sq(self.a)
        for k in range(0, len(self.h)):
            res += norm2sq(self.h[k])
            res += norm2sq(self.v[k])
            res += norm2sq(self.d[k])
        return res


    def norm1(self):
        res = 0
        res += norm1(self.a)
        for k in range(0, len(self.h)):
            res += norm1(self.h[k])
            res += norm1(self.v[k])
            res += norm1(self.d[k])
        return res

    def norm0(self):
        res = 0
        res += (self.a != 0).sum()
        for k in range(0, len(self.h)):
            res += (self.h[k] != 0).sum()
            res += (self.v[k] != 0).sum()
            res += (self.d[k] != 0).sum()
        return res



    def soft_threshold(self, beta): # inplace
        #~ ST = lambda x : np.maximum(np.abs(x)-beta, 0)*np.sign(x)
        #~ self.a = _soft_thresh(self.a, beta) # FIXME : do I have to also ST the app coeffs ?
        L = len(self.h)
        for k in range(0, L):
            self.h[k] = _soft_thresh(self.h[k], beta)
            self.v[k] = _soft_thresh(self.v[k], beta)
            self.d[k] = _soft_thresh(self.d[k], beta)


    def proj_linf(self, beta):
        L = len(self.h)
        self.a = _proj_linf_img(self.a, beta)
        for k in range(0, L):
            self.h[k] = _proj_linf_img(self.h[k], beta)
            self.v[k] = _proj_linf_img(self.v[k], beta)
            self.d[k] = _proj_linf_img(self.d[k], beta)

    def inverse(self):
        tmp = []
        tmp.append(self.a)
        for k in range(0, len(self.h)):
            detcoefs = []
            detcoefs.append(self.h[k])
            detcoefs.append(self.v[k])
            detcoefs.append(self.d[k])
            tmp.append(detcoefs)
        res = pywt.waverec2(tmp, self.wname, mode='per')
        if self.do_random_shifts is True: return _circular_shift(res, -self.shifts[0], -self.shifts[1])
        else: return res














