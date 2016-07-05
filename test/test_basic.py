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

import numpy as np
import unittest

class TestBasic(unittest.TestCase):

    def setUp(self):
        pass

    def test_import(self):
        try:
            import spire
            success = 1
        except ImportError:
            success = 0
        self.assertTrue(success == 1, msg="Failed to import spire for some reason")

    def test_all_imports(self):
        try:
            from spire.operators.tomography import AstraToolbox
            import spire.operators.image
            import spire.operators.misc
            from spire.operators.fft import Fft
            from spire.operators.wavelets import WaveletCoeffs
            from spire.operators.convolution import ConvolutionOperator
            import spire.algorithms.chambollepock
            import spire.algorithms.fista
            import spire.algorithms.conjgrad
            import spire.tomography.sinogram
            import spire.tomography.sirtfilter
            import spire.io
            import spire.utils
            success = 1
        except ImportError:
            success = 0
        self.assertTrue(success == 1, msg="Could not import all modules")

    def test_version(self):
        import spire
        print("Version = %s" % spire.version)

    def tearDown(self):
        pass



def test_suite_basic():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestBasic("test_import"))
    testSuite.addTest(TestBasic("test_all_imports"))
    testSuite.addTest(TestBasic("test_version"))
    return testSuite


def run():
    mysuite = test_suite_basic()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)


if __name__ == '__main__':

    run()











