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
import os
import glob
from spire.io import edf_read, edf_write





def explore_tomo_dataset(folder, file_prefix):

    if not(os.path.isdir(folder)): raise IOError('explore_tomo_dataset(): Not a folder : %s' % folder)
    pref = os.path.join(folder, file_prefix)

    # TODO : more flexible file suffix
    fl = glob.glob(pref + "????.edf")
    fl.sort()
    if fl == []:
        print('Warning: explore_tomo_dataset(): could not find any EDF file in %s matching the file prefix %s' % (folder, file_prefix))
        return {}
    n = len(fl)

    # Read first projection file, assuming the others follow the same format
    proj0 = edf_read(fl[0])
    npx_R, npx_C = proj0.shape

    res = {}
    res['width'] = npx_C
    res['height'] = npx_R
    res['datatype'] = proj0.dtype
    res['number'] = n
    res['files'] = fl

    return res




def get_tomo_file_num(fname):
    """
    get the number of a processed file, for a specific name format.
    Example : get_file_num(dataset_scan_3141.edf) == 3141
    """
    num_str = fname.split('_')[-1].split('.')[0]
    return num_str

#~ TODO :
#~ - Output format other than EDF ; and suffix possibly different from _0123.edf
#~ - It assumes that the data is stored "alphabetically" (data of file_2718.edf comes after data of file_2717.edf)
def apply_processing(myprocessing, dataset, options, extra_args=None):
    """
    Apply a processing on a whole dataset. The processing function is defined by the user.

    Parameters
    ------------
    myprocessing : function
        function taking a numpy.ndarray (and possibly extra_args) as an input, and returning a numpy.ndarray
    dataset : dict
        dataset information returned by explore_tomo_folder()
    folder_out : string
        folder where the results will be saved
    file_prefix_out : string
        file prefix for the reconstruction (eg. ``"rec_"``)
    extra_args : tuple
        (optionnal) extra arguments that should be passed to the user's processing function
    """


    # Arg check
    # ----------
    if not isinstance(dataset, dict):
        raise ValueError('apply_processing(): second argument should be a Python dictionary')
    if not callable(myprocessing):
        raise ValueError('apply_processing(): first argument should be a callable custom function. It has to take an array as an input and return an array.')
    if options and not isinstance(options, dict):
        raise ValueError('apply_processing() : options should be a dictionary')

    # Parse the execution options
    # ----------
    _verbose = True # Verbose is true by default.
    _start = 0
    _number = dataset['number']
    _end = _number -1

    # Turns into a case insensitive search
    options = dict((k.lower(), options[k]) for k in options)
    if options.has_key('verbose'):
        _verbose = True if bool(options['verbose']) else False
    if options.has_key('file_start'): tmp_str1 = get_tomo_file_num(options['file_start'])
    else: tmp_str1 = '0'
    if options.has_key('file_end'): tmp_str2 = get_tomo_file_num(options['file_end'])
    else: tmp_str2 = str(_end)
    try:
        _start = int(tmp_str1)
        _end = int(tmp_str2)
    except ValueError:
        raise ValueError('apply_processing(): options file_start and file_end should be in format "path_0123.edf"') # TODO
    if options.has_key('start'): _start = int(options['start'])
    if options.has_key('end'): _end = int(options['end'])
    if (_start > _end): raise ValueError('apply_processing(): option start or file_start (%d) is greater than end or file_end (%d)' % (_start, _end))
    if (_end > _number): raise ValueError('apply_processing(): option end or file_end (%d) is greater than available files (%d)' % (_end, _number))
    try:
        folder_out = options['output_folder']
        file_prefix_out = options['output_file_prefix']
    except KeyError:
        raise Exception('apply_processing(): please provide "output_folder" and "output_file_prefix" in the options')



    # Processing
    # ----------

    for i in range(_start, _end +1): # TODO : what if the ordering gives a result different from "for f in dataset['files']" ?

        f = dataset['files'][i]
        if _verbose: print("Processing file %s" % os.path.split(f)[-1])

        arr = edf_read(f)
        if extra_args:
            arr_out = myprocessing(arr, extra_args)
        else:
            arr_out = myprocessing(arr)

        fname_out = os.path.join(folder_out, file_prefix_out) + str("%04d" % i) + ".edf"
        edf_write(arr_out, fname_out)
        if _verbose: print("Wrote %s" % fname_out)







