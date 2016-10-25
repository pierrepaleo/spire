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


from setuptools import setup, find_packages
import os
#~ import sys
#~ import glob



def get_version():

    with open("spire/version.py") as fid:
        lines = fid.readlines()
    # Get the line containing "version ="
    idx = -1
    for i, l in enumerate(lines):
        if ("version" in l) and ("=" in l): idx = i
    if idx == -1: raise RuntimeError("Unable to get version from spire/version.py")

    # Extract the version number
    ver = lines[idx].rstrip('\n').split("=")[-1].strip(' ').strip('"')
    return ver



if __name__ == '__main__':

    __version = get_version()

    cmdclass = {}

    packages = ['spire', 'spire.operators', 'spire.tomography', 'spire.algorithms', 'spire.samples', 'spire.third_party']
    package_dir = {"spire": "spire",
            'spire.tomography': 'spire/tomography',
            'spire.algorithms': 'spire/algorithms',
            'spire.operators':'spire/operators',
            'spire.samples': 'spire/samples',
            'spire.third_party': 'spire/third_party'}

    setup(name = "spire",
        version = __version,
        platforms = ["linux_x86", "linux_x86_64"],
        description = "Simple Prototyping for Image Reconstruction",
        author = "Pierre Paleo",
        author_email = "pierre.paleo@esrf.fr",
        maintainer = "Pierre Paleo",
        maintainer_email = "pierre.paleo@esrf.fr",
        url = "https://github.com/pierrepaleo/spire",
        license="BSD",
        #~ packages=find_packages(),
        #~ packages=find_packages(exclude=("test", ),

        packages=packages,
        package_dir = package_dir,
        package_data={'': ['samples/lena.npz', 'samples/brain256.npz', 'samples/brain512.npz', 'samples/brain1024.npz', 'samples/brain2048.npz', 'samples/brain4096.npz']},
        #~ data_files=[("spire-data", ["spire/samples/lena.npz"])], # not copied in site-packages !

        long_description = """
        This module contains various utilities for image processing algorithms based on
        optimization of objective functions involving linear operators.
        It also provides useful wrappers for tomography, FFT, convolution and Wavelet transform.
        """
        )


