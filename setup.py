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


from setuptools import setup, find_packages
import os
#~ import sys
#~ import glob



def get_version():

    with open("vita/version.py") as fid:
        lines = fid.readlines()
    # Get the line containing "version ="
    idx = -1
    for i, l in enumerate(lines):
        if ("version" in l) and ("=" in l): idx = i
    if idx == -1: raise RuntimeError("Unable to get version from vita/version.py")

    # Extract the version number
    ver = lines[idx].rstrip('\n').split("=")[-1].strip(' ').strip('"')
    return ver



if __name__ == '__main__':

    __version = get_version()

    cmdclass = {}

    packages = ['vita', 'vita.operators', 'vita.tomography', 'vita.algorithms', 'vita.samples']
    package_dir = {"vita": "vita",
            'vita.tomography': 'vita/tomography',
            'vita.algorithms': 'vita/algorithms',
            'vita.operators':'vita/operators',
            'vita.samples': 'vita/samples'}

    setup(name = "vita",
        version = __version,
        platforms = ["linux_x86", "linux_x86_64"],
        description = "Various Image processing and Tomography Algorithms",
        author = "Pierre Paleo",
        author_email = "pierre.paleo@esrf.fr",
        maintainer = "Pierre Paleo",
        maintainer_email = "pierre.paleo@esrf.fr",
        url = "https://github.com/pierrepaleo/vita",
        license="BSD",
        #~ packages=find_packages(),
        #~ packages=find_packages(exclude=("test", ),

        packages=packages,
        package_dir = package_dir,
        package_data={'': ['samples/lena.npz']},
        #~ data_files=[("vita-data", ["vita/samples/lena.npz"])], # not copied in site-packages !

        long_description = """
        This module contains various utilities for image processing algorithms based on
        optimization of objective functions involving linear operators.
        It also provides useful wrappers for tomography, FFT, convolution and Wavelet transform.
        """
        )


