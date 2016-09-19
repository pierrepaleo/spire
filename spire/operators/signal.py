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
from math import sqrt, pi


def gradient1d(x):
    """
    Gradient of 1D array
    """
    t1 = np.empty_like(x)
    t2 = np.empty_like(x)
    t1[:-1] = x[1:]
    t1[-1] = 0
    t2[:-1] = x[:-1]
    t2[-1] = 0
    return t1-t2


def div1d(x):
    """
    Divergence of 1D gradient
    """
    t1 = np.empty_like(x)
    t2 = np.empty_like(x)
    t1[:-1] = -x[:-1]
    t1[-1] = 0
    t2[0] = 0
    t2[1:] = x[:-1]
    return t1 + t2



def generate_sine(npts=None, freq=None, nper=None, fs=None, tmax=1.0, offset=0, return_time=False):
    """
    Generate a sine wave: sin(2 pi freq t).
    This can be done with a variety of combinations of the following parameters.

    Parameters
    -----------
    npts: integer
        Number of points of the sine function
    freq: float
        Sine frequency
    nper: float
        Number of visible sine periods
    fs: float
        Sampling frequency
    tmax: float
        The signal is generated between [0, tmax] seconds. Default is tmax = 1.
    offset: float
        phase at the origin. Default is 0.
    return_time: bool
        Return the time vector. Default is False.
    """

    if npts is None:
        if fs is None: raise ValueError("Please provide either number of points (npts) or sampling frequency (fs)")
        npts = tmax*fs

    if freq is None:
        if nper is None: raise ValueError("Please provide either the sine frequency (freq) or the number of visible periods (nper)")
        freq = nper/tmax

    t = np.linspace(0, tmax, npts)
    s = np.sin(2*pi*freq*t + offset)
    if return_time:
        return t, s
    else:
        return s






