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
import random
import string
from math import pi, cos, sin

try:
    import matplotlib.pyplot as plt
    __has_plt__ = True
except ImportError:
    __has_plt__ = False
from math import ceil

# For ImageJ :
import subprocess
import os
from io import tiff_save


# ------------------------------------------------------------------------------
#                       Improved matplotlib image viewer
# ------------------------------------------------------------------------------




def ims(img, cmap=None, legend=None, nocbar=False, share=True):
    """
    image visualization utility.

    img: 2D numpy.ndarray, or list of 2D numpy.ndarray
        image or list of images
    cmap: string
        Optionnal, name of the colorbar to use.
    legend: string, or list of string
        legend under each image
    nocbar: bool
        if True, no colorbar are displayed. Default is False
    share: bool
        if True, the axis are shared between the images, so that zooming in one image
        will zoom in all the corresponding regions of the other images. Default is True
    """
    if not(__has_plt__): raise ImportError("Please install matplotlib to use this function.")
    try:
        _ = img.shape
        nimg = 1
    except AttributeError:
        nimg = len(img)
    #
    if (nimg <= 2): shp = (1,2)
    elif (nimg <= 4): shp = (2,2)
    elif (nimg <= 6): shp = (2,3)
    elif (nimg <= 9): shp = (3,3)
    else: raise ValueError("too many images")
    #
    plt.figure()
    for i in range(nimg):
        curr = list(shp)
        curr.append(i+1)
        curr = tuple(curr)
        if nimg > 1:
            if i == 0: ax0 = plt.subplot(*curr)
            else:
                if share: plt.subplot(*curr, sharex=ax0, sharey=ax0)
                else: plt.subplot(*curr)
            im = img[i]
            if legend: leg = legend[i]
        else:
            im = img
            if legend: leg = legend
        if cmap:
            plt.imshow(im, cmap=cmap, interpolation="nearest")
        else:
            plt.imshow(im, interpolation="nearest")
        if legend: plt.xlabel(leg)
        if nocbar is False: plt.colorbar()

    plt.show()












# ------------------------------------------------------------------------------
#                   Utility to view numpy arrays with imagej
# ------------------------------------------------------------------------------


def _imagej_open(fname):
    # One file
    if isinstance(fname, str):
        cmd = ['imagej', fname]
    # Multiple files
    if isinstance(fname, list):
        cmd = ['imagej'] + fname
    FNULL = open(os.devnull, 'w')
    process = subprocess.Popen(cmd, stdout=FNULL, stderr=FNULL)
    FNULL.close();
    process.wait()
    return process.returncode


def call_imagej(obj):
    # Open file(s)
    if isinstance(obj, str) or (isinstance(obj, list) and isinstance(obj[0], str)):
        return _imagej_open(obj)
    # Open numpy array(s)
    elif isinstance(obj, np.ndarray) or (isinstance(obj, list) and isinstance(obj[0], np.ndarray)):
        if isinstance(obj, np.ndarray):
            data = obj
            if data.dtype == np.float64: data = data.astype(np.float32)
            if data.dtype == np.int64: data = data.astype(np.int32)
            fname = '/tmp/' + _randomword(10) + '.tif'
            tiff_save(fname, data)
            return _imagej_open(fname)
        else:
            fname_list = []
            for i, data in enumerate(obj):
                if data.dtype == np.float64: data = data.astype(np.float32)
                if data.dtype == np.int64: data = data.astype(np.int32)
                fname = '/tmp/' + _randomword(10) + str("_%d.tif" % i)
                fname_list.append(fname)
                tiff_save(fname, data)
            return _imagej_open(fname_list)

    else:
        raise ValueError('Please enter a file name or a numpy array')


def _randomword(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))



# ------------------------------------------------------------------------------
#                               Miscellaneous
# ------------------------------------------------------------------------------



def generate_coords(img_shp, center=None):
    l_r, l_c = float(img_shp[0]), float(img_shp[1])
    R, C = np.mgrid[:l_r, :l_c] # np.indices is faster
    if center is None:
        center0, center1 = l_r / 2., l_c / 2.
    else:
        center0, center1 = center
    R += 0.5 - center0
    C += 0.5 - center1
    return R, C


def ceilpow2(N):
    p = 1
    while p < N:
        p *= 2
    return p


def clip_circle(img, center=None, radius=None):
    R, C = generate_coords(img.shape, center)
    M = R**2+C**2
    res = np.zeros_like(img)
    res[M<radius**2] = img[M<radius**2]
    return res


def ellipse_mask(img_shape, r, c, a, b, phi=None):
    if phi is None: phi = 0
    else: phi = np.deg2rad(phi)
    R, C = generate_coords(img_shape)
    mask = np.zeros(img_shape)
    x = R - r
    y = C - c
    mask[(x*cos(phi)+y*sin(phi))**2/a**2 + (y*cos(phi)-x*sin(phi))**2/b**2 <= 1.] = 1
    return mask





def gaussian1D(sigma):
    ksize = int(ceil(8 * sigma + 1))
    if (ksize % 2 == 0): ksize += 1
    t = np.arange(ksize) - (ksize - 1.0) / 2.0
    g = np.exp(-(t / sigma) ** 2 / 2.0).astype('f')
    g /= g.sum(dtype='f')
    return g


def fftbs(x):
    '''
    Bluestein chirp-Z transform.
    Computes a FFT on a "good" length (power of two) to speed-up the FFT, without extending the spectrum.
    '''
    N = x.shape[0]
    n = np.arange(N)
    b = np.exp((1j*pi*n**2)/N)
    a = x * b.conjugate()
    M = ceilpow2(N) * 2
    #~ A = np.concatenate((a, [0] * (M - N)))
    B = np.concatenate((b, [0] * (M - 2*N + 1), b[:0:-1]))
    C = np.fft.ifft(np.fft.fft(a, M) * np.fft.fft(B))

    c = C[:N]
    return b.conjugate() * c



# not so elegant, but achieves 523 ms vs 3.23 s for (631, 1451)
def fftbs2(img, copy=True, one_axis=False):
    N = img.shape[1]
    n = np.arange(N)
    chirp = np.exp((1j*pi*n**2)/N)
    a = img * chirp.conjugate()
    M = ceilpow2(N) * 2
    chirp2 = np.concatenate((chirp, [0] * (M - 2*N + 1), chirp[:0:-1]))
    a_f = np.fft.fft(a, M, axis=1)
    chirp2_f = np.fft.fft(chirp2)
    C = np.fft.ifft(a_f * chirp2_f, axis=1)
    c = C[:, :N]
    res = chirp.conjugate() * c
    res = res.T
    if copy:
        res = np.ascontiguousarray(res)
    if one_axis: return res
    return fftbs2(res, one_axis=True)











################################################################################
# Phantoms utils. by Alex Opie  <lx_op@orcon.net.nz>
################################################################################


## Copyright (C) 2010  Alex Opie  <lx_op@orcon.net.nz>
##
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.


def phantom(n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
    """
    Create a Shepp-Logan or modified Shepp-Logan phantom.

    A phantom is a known object (either real or purely mathematical)
    that is used for testing image reconstruction algorithms.  The
    Shepp-Logan phantom is a popular mathematical model of a cranial
    slice, made up of a set of ellipses.  This allows rigorous
    testing of computed tomography (CT) algorithms as it can be
    analytically transformed with the radon transform (see the
    function `radon').

    Inputs
    ------
    n : The edge length of the square image to be produced.

    p_type : The type of phantom to produce. Either
    "Modified Shepp-Logan" or "Shepp-Logan". This is overridden
     if `ellipses' is also specified.

    ellipses : Custom set of ellipses to use.  These should be in
    the form
            [[I, a, b, x0, y0, phi],
             [I, a, b, x0, y0, phi],
             ...]
      where each row defines an ellipse.
      I : Additive intensity of the ellipse.
      a : Length of the major axis.
      b : Length of the minor axis.
      x0 : Horizontal offset of the centre of the ellipse.
      y0 : Vertical offset of the centre of the ellipse.
      phi : Counterclockwise rotation of the ellipse in degrees,
      measured as the angle between the horizontal axis and
      the ellipse major axis.
      The image bounding box in the algorithm is [-1, -1], [1, 1],
      so the values of a, b, x0, y0 should all be specified with
      respect to this box.

    Output
    ------
    P : A phantom image.

    Usage example
    -------------
      import matplotlib.pyplot as pl
      P = phantom ()
      pl.imshow (P)

    References
    ----------
    Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
    from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
    Feb. 1974, p. 232.

    Toft, P.; "The Radon Transform - Theory and Implementation",
    Ph.D. thesis, Department of Mathematical Modelling, Technical
    University of Denmark, June 1996.
    """

    if (ellipses is None):
            ellipses = _select_phantom (p_type)
    elif (np.size (ellipses, 1) != 6):
            raise AssertionError ("Wrong number of columns in user phantom")

    # Blank image
    p = np.zeros ((n, n))

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]

    for ellip in ellipses:
            I   = ellip [0]
            a2  = ellip [1]**2
            b2  = ellip [2]**2
            x0  = ellip [3]
            y0  = ellip [4]
            phi = ellip [5] * np.pi / 180  # Rotation angle in radians

            # Create the offset x and y values for the grid
            x = xgrid - x0
            y = ygrid - y0

            cos_p = np.cos (phi)
            sin_p = np.sin (phi)

            # Find the pixels within the ellipse
            locs = (((x * cos_p + y * sin_p)**2) / a2
          + ((y * cos_p - x * sin_p)**2) / b2) <= 1

            # Add the ellipse intensity to those pixels
            p [locs] += I

    return p


def _select_phantom (name):
        if (name.lower () == 'shepp-logan'):
                e = _shepp_logan ()
        elif (name.lower () == 'modified shepp-logan'):
                e = _mod_shepp_logan ()
        else:
                raise ValueError ("Unknown phantom type: %s" % name)

        return e


def _shepp_logan ():
        #  Standard head phantom, taken from Shepp & Logan
        return [[   2,   .69,   .92,    0,      0,   0],
                [-.98, .6624, .8740,    0, -.0184,   0],
                [-.02, .1100, .3100,  .22,      0, -18],
                [-.02, .1600, .4100, -.22,      0,  18],
                [ .01, .2100, .2500,    0,    .35,   0],
                [ .01, .0460, .0460,    0,     .1,   0],
                [ .02, .0460, .0460,    0,    -.1,   0],
                [ .01, .0460, .0230, -.08,  -.605,   0],
                [ .01, .0230, .0230,    0,  -.606,   0],
                [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan ():
        #  Modified version of Shepp & Logan's head phantom,
        #  adjusted to improve contrast.  Taken from Toft.
        return [[   1,   .69,   .92,    0,      0,   0],
                [-.80, .6624, .8740,    0, -.0184,   0],
                [-.20, .1100, .3100,  .22,      0, -18],
                [-.20, .1600, .4100, -.22,      0,  18],
                [ .10, .2100, .2500,    0,    .35,   0],
                [ .10, .0460, .0460,    0,     .1,   0],
                [ .10, .0460, .0460,    0,    -.1,   0],
                [ .10, .0460, .0230, -.08,  -.605,   0],
                [ .10, .0230, .0230,    0,  -.606,   0],
                [ .10, .0230, .0460,  .06,  -.605,   0]]

