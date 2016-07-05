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
from spire.operators.fft import Fft


class TestFft(unittest.TestCase):
    """
    Test the FFT helper around pyfftw :
        - 1D R2C and C2C, forward and inverse
        - 2D R2C and C2C, forward and inverse
        - 3D R2C and C2C, forward and inverse (takes much time)
    """

    def setUp(self):
        # Reproducible results
        np.random.seed(0)
        # Shape and values range
        self.shp = (511, 513, 64)
        self.Range = 10.
        # Error tolerance for tests
        self.maxtol = 1e-7


    def __generate(self, shp, ndims, iscomplex):
        res = np.random.rand(*(shp[0:ndims])) * self.Range
        if iscomplex:
            res = res + 1j*np.random.rand(*(shp[0:ndims])) * self.Range
        return res


    def test_1d_real(self):
        u = self.__generate(self.shp, 1, 0)
        fft = Fft(u)
        uf = fft.fft(u)
        uf0 = np.fft.rfft(u)
        errmax = np.abs(uf - uf0).max()
        self.assertTrue(errmax < self.maxtol, msg="RFFT 1D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))
        u1 = fft.ifft(uf)
        #~ u0 = np.fft.irfft(uf0)
        errmax = np.abs(u1 - u).max() # for odd sizes, numpy.fft.irfft outputs modified shapes
        self.assertTrue(errmax < self.maxtol, msg="IRFFT 1D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))


    def test_1d_complex(self):
        u = self.__generate(self.shp, 1, 1)
        fft = Fft(u)
        uf = fft.fft(u)
        uf0 = np.fft.fft(u)
        errmax = np.abs(uf - uf0).max()
        self.assertTrue(errmax < self.maxtol, msg="FFT 1D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))
        u1 = fft.ifft(uf)
        #~ u0 = np.fft.irfft(uf0)
        errmax = np.abs(u1 - u).max() # for odd sizes, numpy.fft.irfft outputs modified shapes
        self.assertTrue(errmax < self.maxtol, msg="IFFT 1D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))



    def test_2d_real(self):
        u = self.__generate(self.shp, 2, 0)
        fft = Fft(u)
        uf = fft.fft(u)
        uf0 = np.fft.rfft2(u)
        errmax = np.abs(uf - uf0).max()
        self.assertTrue(errmax < self.maxtol, msg="RFFT 2D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))
        u1 = fft.ifft(uf)
        #~ u0 = np.fft.irfft(uf0)
        errmax = np.abs(u1 - u).max() # for odd sizes, numpy.fft.irfft outputs modified shapes
        self.assertTrue(errmax < self.maxtol, msg="IRFFT 2D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))


    def test_2d_complex(self):
        u = self.__generate(self.shp, 2, 1)
        fft = Fft(u)
        uf = fft.fft(u)
        uf0 = np.fft.fft2(u)
        errmax = np.abs(uf - uf0).max()
        self.assertTrue(errmax < self.maxtol, msg="FFT 2D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))
        u1 = fft.ifft(uf)
        #~ u0 = np.fft.irfft(uf0)
        errmax = np.abs(u1 - u).max() # for odd sizes, numpy.fft.irfft outputs modified shapes
        self.assertTrue(errmax < self.maxtol, msg="IFFT 2D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))


    def test_3d_real(self):
        u = self.__generate(self.shp, 3, 0)
        fft = Fft(u)
        uf = fft.fft(u)
        uf0 = np.fft.rfftn(u)
        errmax = np.abs(uf - uf0).max()
        maxtol = 1e-7 # error increase with dimensions
        self.assertTrue(errmax < maxtol, msg="RFFT 3D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))


    def test_3d_complex(self):
        u = self.__generate(self.shp, 3, 1)
        fft = Fft(u)
        uf = fft.fft(u)
        uf0 = np.fft.fftn(u)
        errmax = np.abs(uf - uf0).max()
        maxtol = 1e-7 # error increase with dimensions
        self.assertTrue(errmax < maxtol, msg="FFT 3D failed on shape %s (err_max = %e when it should be < %e)" % (str(u.shape), errmax, self.maxtol))



    def tearDown(self):
        pass




#~ if __name__ == "__main__":
    #~ unittest.main()


def test_suite_all_fft():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestFft("test_1d_real"))
    testSuite.addTest(TestFft("test_1d_complex"))
    testSuite.addTest(TestFft("test_2d_real"))
    testSuite.addTest(TestFft("test_2d_complex"))
    #~ testSuite.addTest(TestFft("test_3d_real")) # takes too much time
    #~ testSuite.addTest(TestFft("test_3d_complex")) # takes too much time
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_fft()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)



