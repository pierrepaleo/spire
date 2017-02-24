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
try:
    from scipy.ndimage import filters
    __has_ndimage__ = True
except ImportError:
    __has_ndimage__ = False

def convolve_sep_scipy(img, kernel, mode):
    res = filters.convolve1d(img, kernel, axis= -1, mode=mode)
    return filters.convolve1d(res, kernel, axis=0, mode=mode)


def convolve2_scipy(img, kernel, mode):
    return filters.convolve(img, kernel, mode=mode)


def _fftconv1(im, k):
    """
    FFT based convolution with 'reflect' boundary condition, axis 1.
    This is not optimized at all ; scipy should be used for small kernel sizes
    """
    Nr, Nc = im.shape
    s = (k.shape[0] - 1)/2
    im2 = np.zeros((Nr, 3*Nc))
    im2[:, :Nc] = im[:, ::-1]
    im2[:, Nc:2*Nc] = im
    im2[:, 2*Nc:] = im[:, ::-1]
    return np.fft.ifft(np.fft.fft(im2, axis=1) * np.fft.fft(k, 3*Nc), axis=1)[:, Nc+s:2*Nc+s].real

def _fftconv0(im, k):
    """
    FFT based convolution with 'reflect' boundary condition, axis 0
    This is not optimized at all ; scipy should be used for small kernel sizes
    """
    Nr, Nc = im.shape
    s = (k.shape[0] - 1)/2
    im2 = np.zeros((3*Nr, Nc))
    im2[:Nr, :] = im[::-1, :]
    im2[Nr:2*Nr, :] = im
    im2[2*Nr:, :] = im[::-1, :]
    return np.fft.ifft(np.fft.fft(im2, axis=0) * np.fft.fft(k, 3*Nr), axis=0)[Nr+s:2*Nr+s, :].real

def convolve_sep_fft(img, kernel, mode):
    res = _fftconv1(img, kernel)
    return _fftconv0(res, kernel)



# TODO : convolution along a given axis
class ConvolutionOperator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.is2D = True if len(kernel.shape) > 1 else False
        self.mode = 'reflect' #{'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        if __has_ndimage__:
            self.convolve_sep = convolve_sep_scipy
            self.convolve2 = convolve2_scipy
        else:
            self.convolve_sep = convolve_sep_fft
            self.convolve2 = convolve2_fft
            if self.mode != 'reflect':
                raise NotImplementedError('ConvolutionOperator: please install scipy for using mode %s' % (self.mode))

    def __mul__(self, img):
        '''
        do the actual convolution.
        If the kernel is 1D, a separable convolution is done.
        '''
        if self.is2D:
            return self.convolve2(img, self.kernel, self.mode)
        else:
            return self.convolve_sep(img, self.kernel, self.mode)



    def adjoint(self):
        res = ConvolutionOperator(self.kernel)
        res.kernel = res.kernel.T
        return res






