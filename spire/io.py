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
Tools for (very) basic I/O.
I/O is not the primary purpose of this package, so it would be better using another
external module for proper read/write !
"""

from __future__ import division
import numpy as np

# EDF
from .third_party import EdfFile
# tiff
from .third_party import tifffile

# HDF5
try:
    import h5py
    __has_hdf5__ = True
except ImportError:
    __has_hdf5__ = False
# mat (old versions)
try:
    import scipy.io
    __has_scipyio__ = True
except ImportError:
    __has_scipyio__ = False



def edf_read(fname, numframe=0):
    '''
    Read a EDF file and store it into a numpy array
    '''
    fid = EdfFile.EdfFile(fname)
    data = fid.GetData(numframe)
    fid.File.close()
    return data


def edf_write(data, fname, info=None):
    '''
    Save a numpy array into a EDF file
    '''
    if info is None: info = {}
    edfw = EdfFile.EdfFile(fname, access='w+') # Overwrite !
    edfw.WriteImage(info, data)
    edfw.File.close()



def h5_read(fname, numframe=0):
    fid = h5py.File(fname)
    res = fid[fid.keys()[numframe]].value
    fid.close()
    return res


def h5_write(arr, fname):
    raise NotImplementedError('H5 write is not implemented yet, please do it manually with h5py')


def loadmat(fname, framenum=0):
    try:
        res = scipy.io.loadmat(fname)
    except NotImplementedError: # Matlab >= 7.3 files
        res = h5_read(fname, framenum)
    return res


def tiff_read(fname, *args, **kwargs):
    return tifffile.imread(fname, *args, **kwargs)


def tiff_save(fname, data, **kwargs):
    return tifffile.imsave(fname, data, **kwargs)


def tiff_write(fname, data, **kwargs):
    return tifffile.imsave(fname, data, **kwargs)








