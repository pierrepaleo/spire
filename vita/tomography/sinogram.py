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
import numpy as np
from vita.utils import generate_coords
from vita.algorithms.simplex import _minimize_neldermead
from math import pi
from vita.operators.fft import Fft

# ------------------------------------------------------------------------------
# ------------Cupping reduction : "sinogram straightening" ---------------------
# ------------------------------------------------------------------------------


def straighten_sino(sino, order=3):
    n_angles, n_pix = sino.shape
    x = np.arange(n_pix)
    sino_corr = np.zeros_like(sino)

    i = 0
    for line in range(n_angles):
        y = sino[line, :]
        # Least-Squares, 3rd order polynomial :
        #~ X = np.array([np.ones(n_pix), x, x**2, x**3]).T
        #~ X = X[:, ::-1] # numpy convention
        #~ z0 = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        z = np.polyfit(x, y, order)
        f = np.poly1d(z)
        sino_corr[line, :] = y - f(x)
    return sino_corr





# ------------------------------------------------------------------------------
# ------------ Determine the center of rotation --------------------------------
# ------------------------------------------------------------------------------

def _gaussian_kernel(ksize, sigma):
    '''
    Creates a gaussian function of length "ksize" with std "sigma"
    '''
    x = np.arange(ksize) - (ksize - 1.0) / 2.0
    gaussian = np.exp(-(x / sigma) ** 2 / 2.0).astype(np.float32)
    #gaussian /= gaussian.sum(dtype=np.float32)
    return gaussian


def get_center_shifts(sino, smin, smax, sstep=1):
    '''
    Determines the center of rotation according to
        Vo NT Et Al, "Reliable method for calculating the center of rotation in parallel-beam tomography", Opt Express, 2014

    The idea is the following :
        - create a reflected version of the original sinogram
        - shift this mirrored sinogram and append it to the original one
        - take the Fourier Transform and see what happens in the vertical line (u = 0)
        - repeat for different shifts
    Please note that this method is quite slow for large sinograms

    @param sino : sinogram as a numpy array
    @param smin: minimum shift of lower sinogram (can be negative)
    @param smax: maximum shift of lower sinogram
    @param sstep: shift step (can be less than 1 for subpixel precision)
    '''

    if sstep < 1: raise NotImplementedError('subpixel precision is not implemented yet...')
    sino_flip = sino[::-1, :]
    n_angles, n_px = sino.shape
    radius = n_px/8. #n_px #small radius => big complement of double-wedge
    s_vec = np.arange(smin, smax+1, sstep)*1.0
    Q_vec = np.zeros_like(s_vec)
    for i,s in enumerate(s_vec):
        print("[calc_center_shifts] Case %d/%d" % (i+1,s_vec.shape[0]))
        # Create the artificial 360° sinogram (cropped)
        sino2 = np.zeros((2*n_angles, n_px - abs(s)))
        if s > 0:
            sino2[:n_angles, :] = sino[:, s:]
            sino2[n_angles:, :] = sino_flip[:, :-s]
        elif s < 0:
            sino2[:n_angles, :] = sino[:, :s]
            sino2[n_angles:, :] = sino_flip[:, -s:]
        else:
            sino2[:n_angles, :] = sino
            sino2[n_angles:, :] = sino_flip

    #figure(); imshow(sino2);

        # Create the mask "outside double wedge" (see [1])
        R, C = generate_coords(sino2.shape)
        mask = 1 - (np.abs(R) <= np.abs(C)*radius)

        # Take FT of the sinogram and compute the Fourier metric
        fft = Fft(sino2, force_complex=True)
        sino2_f = fft.fft(sino2)
        sino_f = np.abs(np.fft.fftshift(sino2_f))

    #figure(); imshow(np.log(1+sino_f) * mask, interpolation="nearest"); colorbar();

        #~ sino_f = np.log(sino_f)
        Q_vec[i] = np.sum(sino_f * mask)/np.sum(mask)

    s0 = s_vec[Q_vec.argmin()]
    return n_px/2 + s0/2 - 0.5






def centroid_objective(X, n_angles, centr):
    """
    Helper function for get_center()
    """
    offs, amp, phi = X
    t = np.linspace(0, n_angles, n_angles)
    _sin = offs + amp * np.sin(2*pi*(1./(2*n_angles))*t + phi)
    return np.sum((_sin - centr)**2)





def get_center(sino, debug=False):
    '''
    Determines the center of rotation of a sinogram by computing the center of gravity of each row.
    The array of centers of gravity is fitted to a sine function.
    The axis of symmetry is the estimated center of rotation.
    '''

    n_a, n_d = sino.shape
    # Compute the vector of centroids of the sinogram
    i = range(n_d)
    centroids = np.sum(sino*i, axis=1)/np.sum(sino, axis=1)

    # Fit with a sine function : phase, amplitude, offset.
    # Uses Nelder-Mead downhill-simplex algorithm
    cmax, cmin = centroids.max(), centroids.min()
    offs = (cmax + cmin)/2.
    amp = (cmax - cmin)/2.
    phi = 1.1 # !
    x0 = (offs, amp, phi)
    sol, _energy, _iterations, _success, _msg = _minimize_neldermead(centroid_objective, x0, args=(n_a, centroids))

    offs, amp, phi = sol

    if debug:
        t = np.linspace(0, n_a, n_a)
        _sin = offs + amp * np.sin(2*pi*(1./(2*n_a))*t + phi)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(centroids); plt.plot(_sin)
        plt.show()

    return offs


# ------------------------------------------------------------------------------
# (------------ Sinogram consistency conditions --------------------------------
# ------------------------------------------------------------------------------



def consistency(sino, order=0):
    """
    Measure the "sinogram consistency" by checking the Helgason-Ludwig (HL) condition :

    .. math::

        \int_0^\pi \int_{-\infty}^\infty s^n e^{j k \theta} p(\theta, s) \diff s d \theta \; = \, 0

    for k > n >= 0 and k - n even.

    order: integer
        Order of the HL condition
    """
    raise NotImplementedError("Not implemented yet")





# ------------------------------------------------------------------------------
# --------------------------------- Denoising ----------------------------------
# ------------------------------------------------------------------------------

def denoise_sg(sino, ftype=None, order=5):
    """
    Sinogram denoising with Savitzky-Golay filtering.

    sino: numpy.ndarray
        input sinogram
    ftype: string
        "cubic" or "quintic"
    order: integer
        number of points, can be 5, 7, 9 for cubic and 7, 9 for quintic
    """
    c5 = np.array([-3., 12, 17, 12, -3])/35.
    c7 = np.array([-2., 3, 6, 7, 6, 3, -2])/21.
    c9 = np.array([-21., 14, 39, 54, 59, 54, 39, 14, -21])/231.
    q7 = np.array([5., -30, 75, 131, 75, -30, 5])/231.
    q9 = np.array([15., -55, 30, 135, 179, 135, 30, -55, 15])/429.

    if ftype is None: ftype = "cubic"
    if ftype == "cubic":
        if order == 5: c = c5
        elif order == 7: c = c7
        elif order == 9: c = c9
        else: raise NotImplementedError()
    elif ftype == "quintic":
        if order == 7: c = q7
        elif order == 9: c = q9
        else: raise NotImplementedError()
    else: raise ValueError("denoise_sg(): unknown filtering type %s" % ftype)

    res = np.zeros_like(sino)
    # TODO : improve with a "axis=1 convolution"
    for i in range(sino.shape[0]):
        res[i] = np.convolve(sino[i], c, mode="same")
    return res

    # For general case :
    #~ x = np.arange(npts//2, npts//2, npts)
    #~ XT = np.vstack((x**0, x**1, x**2, x**3)) # and so on, for degree
    #~ sol = np.linalg.inv(XT.dot(XT.T)).dot(XT)
    #~ c0 = sol[0, :] # smoothing
    #~ c1 = sol[1, :] # first derivative





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
#           Rings artifacts correction : Münch Et Al. algorithm
# Copyright (c) 2013, Elettra - Sincrotrone Trieste S.C.p.A.
# All rights reserved.
# ------------------------------------------------------------------------------
try:
    import pywt
    __has_pywt__ = True
except ImportError:
    __has_pywt__ = False
    print("Warning: pywt package not found, cannot use Fourier-Wavelet destriping")


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


