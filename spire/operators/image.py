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
from scipy.ndimage import convolve


def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    Code from https://github.com/emmanuelle/tomo-tv/
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


def div(grad):
    '''
    Compute the divergence of a gradient
    Code from https://github.com/emmanuelle/tomo-tv/
    '''
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


def gradient_axis(x, axis=-1):
    '''
    Compute the gradient (keeping dimensions) along one dimension only.
    By default, the axis is -1 (diff along columns).
    '''
    t1 = np.empty_like(x)
    t2 = np.empty_like(x)
    if axis != 0:
        t1[:, :-1] = x[:, 1:]
        t1[:, -1] = 0
        t2[:, :-1] = x[:, :-1]
        t2[:, -1] = 0
    else:
        t1[:-1, :] = x[1:, :]
        t1[-1, :] = 0
        t2[:-1, :] = x[:-1, :]
        t2[-1, :] = 0
    return t1-t2



def div_axis(x, axis=-1):
    '''
    Compute the opposite of divergence (keeping dimensions), adjoint of the gradient, along one dimension only.
    By default, the axis is -1 (div along columns).
    '''
    t1 = np.empty_like(x)
    t2 = np.empty_like(x)
    if axis != 0:
        t1[:, :-1] = -x[:, :-1]
        t1[:, -1] = 0
        t2[:, 0] = 0
        t2[:, 1:] = x[:, :-1]
    else:
        t1[:-1,: ] = -x[:-1, :]
        t1[-1, :] = 0
        t2[0, :] = 0
        t2[1:, :] = x[:-1, :]
    return t1 + t2





def huber(x, mu):
    """
    Huber function, i.e approximation of the absolute value
    """
    res = np.zeros_like(x)
    M = np.abs(x)
    M1 = M < mu
    M2 = np.logical_not(M1)
    res[M1] = x[M1]**2 / (2*mu) + mu/2.
    res[M2] = M[M2]
    return res

def psi(x, mu):
    '''
    Huber function needed to compute tv_smoothed
    '''
    res = np.abs(x)
    m = res < mu
    #~ if (m.sum() > 1): print(m.sum()) # Debug
    res[m] = x[m]**2/(2*mu) + mu/2
    return res


def tv_smoothed(x, mu):
    '''
    Moreau-Yosida approximation of Total Variation
    see Weiss, Blanc-Féraud, Aubert, "Efficient schemes for total variation minimization under constraints in image processing"
    '''
    g = gradient(x)
    g = np.sqrt(g[0]**2 + g[1]**2)
    return np.sum(psi(g, mu))


#~ def grad_tv_smoothed(x, mu):
    #~ '''
    #~ Gradient of Moreau-Yosida approximation of Total Variation
    #~ '''
    #~ g = gradient(x)
    #~ g_mag = np.sqrt(g[0]**2 + g[1]**2)
    #~ m = g_mag >= mu
    #~ m2 = (m == False) #bool(1-m)
    #~ #if (m2.sum() > 30): print(m2.sum()) # Debug
    #~ g[0][m] /= g_mag[m]
    #~ g[1][m] /= g_mag[m]
    #~ g[0][m2] /= mu
    #~ g[1][m2] /= mu
    #~ return -div(g)


def grad_tv_smoothed(x, mu):
    '''
    Gradient of Moreau-Yosida approximation of Total Variation
    '''
    g = gradient(x)
    m = np.maximum(mu, np.sqrt(g[0]**2 + g[1]**2))
    #~ if (m2.sum() > 30): print(m2.sum()) # Debug
    g /= m
    return -div(g)



def proj_l2(g, Lambda=1.0):
    '''
    Proximal operator of the L2,1 norm (for isotropic TV)

    .. math::

        L_{2,1}(u) = \sum_i \left\|u_i\right|_2

    i.e pointwise projection onto the L2 unit ball.

    Parameters
    ------------
    g : gradient-like numpy array
    Lambda : magnitude of the unit ball
    '''
    res = np.copy(g)
    n = np.maximum(np.sqrt(np.sum(g**2, 0))/Lambda, 1.0)
    res[0] /= n
    res[1] /= n
    return res


def proj_l2_img(img, Lambda=1.0):
    '''
    Proximal operator of the L2,1 norm (for isotropic TV)

    .. math::

        L_{2,1}(u) = \sum_i \left\|u_i\right|_2

    i.e pointwise projection onto the L2 unit ball.

    Parameters
    -----------
    g : 2D numpy array
    Lambda : magnitude of the unit ball
    '''
    res = np.copy(img)
    n = np.maximum(np.abs(img)/Lambda, 1.0)
    res /= n
    return res




def proj_linf(x, Lambda=1.):
    '''
    Proximal operator of the dual of L1 norm (for anisotropic TV),
    i.e pointwise projection onto the L-infinity unit ball.

    x : variable
    Lambda : radius of the L-infinity ball
    '''
    return np.minimum(np.abs(x), Lambda) * np.sign(x)



# ------------------------------------------------------------------------------
# ------------------------------ Norms -----------------------------------------
# ------------------------------------------------------------------------------


def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())


def norm1(mat):
    return np.sum(np.abs(mat))


def dot(mat1, mat2):
    return np.dot(mat1.ravel(), mat2.ravel())


def entropy(img):
    '''
    Computes the entropy of an image (similar to Matlab function)
    '''
    h, _ = np.histogram(img, 256)
    h = h.astype('f')
    h /= 1.0*img.size
    h[h == 0] = 1.0
    return -np.sum(h*np.log(h))


def KL(img1, img2):
    '''
    Computes the Kullback-Leibler divergence between two images
    Mind that this function is not symmetric. The second argument should be the "reference" image.
    '''
    x, _ = np.histogram(img1, 256)
    y, _ = np.histogram(img2, 256)
    m = (y != 0) # integers
    x_n, y_n = x[m], y[m]
    m = (x_n != 0)
    x_n, y_n = 1.0 * x_n[m], 1.0 * y_n[m]
    Sx, Sy = x.sum()*1.0, y.sum()*1.0
    return (1.0/Sx) * np.sum(x_n * np.log(x_n/y_n * Sy/Sx))


# ------------------------------------------------------------------------------
# ------------------------- Transforms -----------------------------------------
# ------------------------------------------------------------------------------

def anscombe(x):
    """
    Anscombe transform, a variance stabilizing function.
    Maps a Poisson-distributed data to a Gaussian-distributed data
    """
    return 2*np.sqrt(x + 3.0/8)


def ianscombe(y):
    """
    Inverse of the Anscombe transform
    """
    return (y**2)/4. + sqrt(3/2.)/4./y - 11./8./(y**2) + 5/8.*sqrt(3./2)/(y**3) -1/8.



def ibarlett(x):
    """
    Barlett inverse VST, more numerically stable than the inverse Anscombe VST
    """
    return x**2 / 4. - 3./8




# ------------------------------------------------------------------------------
# ---------------------------- Misc --------------------------------------------
# ------------------------------------------------------------------------------

def expand_reflect(img, rows_ext, cols_ext):
    """
    Create an image with an extended support.
    The borders are reflected in the extended image.

    rows_ext and cols_ext should be even.
    """
    Nr, Nc = img.shape
    Nr2, Nc2 = Nr+rows_ext, Nc+cols_ext
    img2 = np.zeros((Nr2, Nc2))

    """
    *---------------*
    |               |
    |   a*-----*d   |
    |    |     |    |
    |    |     |    |
    |   b*-----*c   |
    |               |
     *--------------*
    """
    hr = rows_ext//2
    hc = cols_ext//2
    a = (hr, hc)
    b = (hr+Nr, hc)
    c = (hr+Nr, hc+Nc)
    d = (hr, hc+Nc)
    # inner
    img2[a[0]:b[0], a[1]:d[1]] = np.copy(img) # TODO: odd sizes
    # left
    img2[a[0]:b[0], a[1]-hc:a[1]] = np.copy(img[:, :hc][:, ::-1])
    # bottom
    img2[b[0]:b[0]+hr, b[1]:c[1]] = np.copy(img[-hr:, :][::-1, :])
    # right
    img2[d[0]:c[0], d[1]:d[1]+hc] = np.copy(img[:, -hc:][:, ::-1])
    # top
    img2[a[0]-hr: a[0], a[1]:d[1]] = np.copy(img[:hr, :][::-1, :])
    # top-left
    img2[a[0]-hr:a[0], a[1]-hc:a[1]] = np.copy(img[:hr, :hc][::-1, ::-1])
    # bottom-left
    img2[b[0]:b[0]+hr, b[1]-hc:b[1]] = np.copy(img[-hr:, :hc][::-1, ::-1])
    # bottom-right
    img2[c[0]:c[0]+hr, c[1]:c[1]+hc] = np.copy(img[-hr:, -hc:][::-1, ::-1])
    # top-right
    img2[d[0]-hr: d[0], d[1]:d[1]+hc] = np.copy(img[:hr, -hc:][::-1, ::-1])

    return img2


def expand_reflect2(img, borderwidth):
    """
    Create an image with twice the support of the input image.
    The borders are reflected in the extended image.
    """
    Nr, Nc = img.shape
    Nr2, Nc2 = 2*Nr, 2*Nc
    img2 = np.zeros((Nr2, Nc2))

    """
    *---------------*
    |               |
    |   a*-----*d   |
    |    |     |    |
    |    |     |    |
    |   b*-----*c   |
    |               |
     *--------------*
    """
    a = (Nr//2, Nc//2)
    b = (Nr//2+Nr, Nc//2)
    c = (Nr//2+Nr, Nc//2+Nc)
    d = (Nr//2, Nc//2+Nc)
    w = borderwidth
    # inner
    img2[a[0]:b[0], a[1]:d[1]] = np.copy(img)
    # left
    img2[a[0]:b[0], a[1]-w:a[1]] = np.copy(img[:, :w][:, ::-1])
    # bottom
    img2[b[0]:b[0]+w, b[1]:c[1]] = np.copy(img[-w:, :][::-1, :])
    # right
    img2[d[0]:c[0], d[1]:d[1]+w] = np.copy(img[:, -w:][:, ::-1])
    # top
    img2[a[0]-w: a[0], a[1]:d[1]] = np.copy(img[:w, :][::-1, :])
    # top-left
    img2[a[0]-w:a[0], a[1]-w:a[1]] = np.copy(img[:w, :w][::-1, ::-1])
    # bottom-left
    img2[b[0]:b[0]+w, b[1]-w:b[1]] = np.copy(img[-w:, :w][::-1, ::-1])
    # bottom-right
    img2[c[0]:c[0]+w, c[1]:c[1]+w] = np.copy(img[-w:, -w:][::-1, ::-1])
    # top-right
    img2[d[0]-w: d[0], d[1]:d[1]+w] = np.copy(img[:w, -w:][::-1, ::-1])

    return img2













def scale_minmax(data, vmin=0, vmax=1, clip=False):
    """
    Scale the data values between [vmin, vmax].

    Parameters
    -----------
    data: numpy.ndarray
        Data to scale the values from
    vmin: float, optional
        Minimum value of the output range
    vmax: float, optional
        Maximum value of the output range
    clip: bool, optional
        Force the data to lie in [vmin, vmax] after operations
    """
    if vmin > vmax: raise ValueError("Must have vmin <= vmax")

    Ai = data.min()
    Bi = data.max()
    Ao = vmin
    Bo = vmax
    alpha = (Bo-Ao)/(Bi-Ai)
    beta = Ao - alpha*Ai
    res = alpha*data + beta
    if clip: res = np.clip(res, vmin, vmax)
    return res






from scipy.ndimage import convolve
def estimate_noise_std(img):
    """
    Given a noisy image, estimate the variance of the noise,
    assuming that the noise is additive and Gaussian.

    Reference
    ----------
    Fast Noise Variance Estimation
    COMPUTER VISION AND IMAGE UNDERSTANDING
    Vol. 64, No. 2, September, pp. 300–302, 1996
    ARTICLE NO 0060
    """
    H, W = img.shape
    kern = np.array([[1, -2, 1.],[-2,4,-2],[1,-2,1]])
    i2 = np.abs(convolve(img, kern))
    return sqrt(pi/2)*np.sum(i2)/(6*(W-2)*(H-2))









