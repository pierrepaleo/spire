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

"""
Simple Implementation of the "SIRT Filter" from [1].

[1] Pelt, D.M., & De Andrade, V. (2017).
Improved tomographic reconstruction of large-scale real-world data by filter optimization. Advanced Structural and Chemical Imaging 2: 17.
http://rdcu.be/niW6


TODO:
- Fix the reconstruction with convolution in Fourier domain (for now only reconst_dummy is working)
- Use the "strip" projector model, preferably with supersampling, to minimize the shift invariance error of (P^T P)
"""






from __future__ import division
import numpy as np
from spire.operators.tomography import AstraToolbox
from spire.operators.image import gradient, div
import os
try:
    import h5py
    __has_h5py__ = True
except ImportError:
    print("Warning: SirtFilter: h5py not found, filters cannot be saved in HDF5 format")
    __has_h5py__ = False


def _str_implode(string_list, separator):
    return separator.join(string_list)

def _ceilpow2(N):
    p = 1
    while p < N:
        p *= 2
    return p


def _open(fname, fmt):
    if fmt == '.npz':
        f_desc = np.load(fname)
        f_data = f_desc['data']
        f_nx = f_desc['nx']
        f_ny = f_desc['ny']
        f_ndet = f_desc['ndet']
        f_nproj = f_desc['nproj']
        f_iter = f_desc['iterations']
        f_Lambda = f_desc['Lambda']
    elif fmt == '.h5':
        f_desc = h5py.File(fname, 'r')
        d_desc = f_desc['sirtfilter']
        f_data = d_desc.value
        f_nx = d_desc.attrs['nx']
        f_ny = d_desc.attrs['ny']
        f_ndet = d_desc.attrs['ndet']
        f_nproj = d_desc.attrs['nproj']
        f_iter = d_desc.attrs['iterations']
        f_Lambda = d_desc.attrs['Lambda']
    f_desc.close()
    return f_data, f_nx, f_ny, f_ndet, f_nproj, f_iter, f_Lambda


def _save(fname, data, n_x, n_y, n_det, nAng, niter, Lambda=0):
    fmt = os.path.splitext(fname)[-1]
    if fmt == '.npz':
        np.savez_compressed(fname, data=data,
                            nx=np.int32(n_x),
                            ny=np.int32(n_y),
                            ndet=np.int32(n_det),
                            nproj=np.int32(nAng),
                            iterations=np.int32(niter),
                            Lambda=np.float32(Lambda))
    elif fmt == '.h5':
        f_desc = h5py.File(fname, 'w')
        d_desc = f_desc.create_dataset('sirtfilter', data=data)
        d_desc.attrs['nx'] = np.int32(n_x)
        d_desc.attrs['ny'] = np.int32(n_y)
        d_desc.attrs['ndet'] = np.int32(n_det)
        d_desc.attrs['nproj'] = np.int32(nAng)
        d_desc.attrs['iterations'] = np.int32(niter)
        d_desc.attrs['Lambda'] = np.float32(Lambda)
        f_desc.close()




def _convolve(sino, thefilter):
    npx = sino.shape[1]
    sz = _ceilpow2(npx)*2
    sino_f = np.fft.fft(sino, sz, axis=1) * thefilter
    return np.fft.ifft(sino_f , axis=1)[:, :npx].real

def _convolve_dummy(sino, filt):
    # Convolution in direct space
    # Inefficient as both supports are large, used for debug
    res = np.zeros_like(sino)
    for i in range(sino.shape[0]):
        res[i] = np.convolve(sino[i], filt[i], mode="same")
    return res



def _compute_filter_operator(n_x, n_y, P, PT, alph, n_it, lambda_tikhonov=0):
        x = np.zeros((n_y, n_x), dtype=np.float32)
        x[n_y//2, n_x//2] = 1
        xs = np.zeros_like(x)
        for i in range(n_it):
            xs += x
            x -= alph*PT(P(x))
            # Laplacian reg.
            #~ if lambda_tikhonov != 0: x += alph*lambda_tikhonov*div(gradient(x))
            # Identity reg.
            if lambda_tikhonov != 0: x -= alph*lambda_tikhonov*x
            if ((i+1) % 10 == 0): print("Iteration %d / %d" % (i+1, n_it))
        return xs




class SirtFilter(object):
    def __init__(self, tomo, n_it, savedir=None, lambda_tikhonov=0, hdf5=False, super_sampling=4):
        '''
        Initialize the SIRT-Filter class.

        tomo: AstraToolbox instance
            tomography configuration the SirtFilter will be based on
        n_it : integer
            number of iterations for the SIRT algorithm
        savedir : string
            Folder where the filter will be stored
        lambda_tikhonov: float
            regularization parameter for a Tikhonov (L2-squared) regularization
        hdf5: bool
            Use True if you want to store the filter as an HDF5 file (rather than npz)
        '''

        self.tomo = tomo
        self.n_it = n_it
        self.hdf5 = hdf5
        self.thefilter = self._compute_filter(savedir=savedir, super_sampling=super_sampling, lambda_tikhonov=lambda_tikhonov)

    def _compute_filter(self, savedir=None, super_sampling=4, lambda_tikhonov=0):

        n_x = self.tomo.n_x
        n_y = self.tomo.n_y
        n_det = self.tomo.dwidth
        nAng = self.tomo.n_a
        niter = self.n_it

        # Check if filter is already calculated for this geometry
        if savedir is not None:
            if not(os.path.isdir(savedir)): raise Exception('%s no such directory' % savedir)
            fmt = '.npz' if not self.hdf5 else '.h5'
            if not(__has_h5py__) and self.hdf5:
                print("Warning: SirtFilter: HDF5 format requestred although h5py is not available. Filter will be exported into .npz format.")
                fmt = '.npz'
            fname = _str_implode(['sirtfilter', str(n_x), str(n_y), str(nAng), str(niter)], '_') + fmt
            fname = os.path.join(savedir, fname)
            if os.path.isfile(fname):
                f_data, f_nx, f_ny, f_ndet, f_nproj, f_iter, f_Lambda = _open(fname, fmt)
                # CHECKME : nx and ny are checked up to +- 1 (since 1 pixel is added for even shape)
                if ((f_nx - n_x) > 1) or ((f_ny - n_y) > 1) or f_nproj != nAng or f_iter != niter or f_Lambda != lambda_tikhonov:
                    print('Warning : file %s does not match the required geometry or number of iterations. Re-computing the filter' % fname)
                else:
                    print('Loaded %s' % fname)
                    return f_data
            else:
                print('Filter %s not found. Computing the filter.' % fname)


        nDet = self.tomo.dwidth
        alph = 1./(nAng*nDet)

        # Always use an odd number of detectors (to be able to set the center pixel to one)
        size_increment = ((n_x & 1) | (n_y & 1))
        if (n_x % 2) == 0:
            n_x += 1
        if (n_y % 2) == 0:
            n_y += 1

        # Initialize ASTRA with this new geometry
        AST2 = AstraToolbox((n_x, n_y), nAng, super_sampling=super_sampling, rot_center=self.tomo.rot_center) # rot center is not required in the computation of the filter
        P = lambda x : AST2.proj(x) #*3.14159/2.0/nAng
        PT = lambda y : AST2.backproj(y, filt=False)

        # Compute the filter with this odd shape
        xs = _compute_filter_operator(n_x, n_y, P, PT, alph, niter, lambda_tikhonov=lambda_tikhonov)

        # The filtering is done in the sinogram domain, using FFT
        # The filter has to be forward projected, then FT'd

        # Forward project
        filter_projected = alph*P(xs)
        # Save the projected filter
        self.filter_projected = filter_projected

        #
        # FIXME: the Fourier convolution is not working for now.
        # As a workaround, just use reconst_dummy instead of reconst.
        #

        # The convolution theorem states that the size of the FFT should be
        # at least 2*N-1  where N is the original size.
        # Here we make both slice (horizontal size N in real domain) and filter (possibly N+1 in real domain)
        # have a size nextpow2(N)*2 in Fourier domain, which should provide good performances for FFT.
        nexpow = _ceilpow2(nDet)
        npix = max(n_x, n_y) + size_increment

        # Manual fftshift
        filter_projected_zpad = np.zeros((nAng, 2*nexpow), dtype=np.float32)
        # Half right (+1) goes to the left
        filter_projected_zpad[:, 0:nDet//2 + size_increment] = filter_projected[:, nDet//2:nDet]
        # Half left goes to the right
        filter_projected_zpad[:, -nDet//2:] = filter_projected_zpad[:, 1:nDet//2 + (1+size_increment)][:, ::-1]
        # FFT
        f_fft = np.fft.fft(filter_projected_zpad, axis=1)

        #~ plot(filter_projected_zpad[30])
        #~ figure(); plot(np.abs(f_fft[30]))


        # Result should be real, since it will be multiplied and backprojected.
        # With the manual zero-padding, the filter is real and symmetric, so its Fourier
        # Transform is also real and symmetric.
        result = f_fft.real.astype(np.float32) # Beware of FFT changing type !
        # Actually only the half of this can be stored (symmetric filter).
        # Take fft.rfft, or fft.fft[:, len/2+1]

        if savedir is not None:
            _save(fname, result, n_x, n_y, nDet, nAng, niter, lambda_tikhonov)
        return result


    def reconst(self, sino):
        s = _convolve(sino, self.thefilter)
        return self.tomo.backproj(s, filt=False)

    def reconst_dummy(self, sino):
        s = _convolve_dummy(sino, self.filter_projected)
        return self.tomo.backproj(s, filt=False)


