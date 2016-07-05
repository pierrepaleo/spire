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
import spire.operators as operators
import spire.operators.misc
import spire.operators.image

class TestOperators(unittest.TestCase):

    def setUp(self):
        # Reproducible results
        np.random.seed(0)

    def test_check_adjoint(self):
        """
        Test if the check_adjoint() function is reliable
        """

        # 2D operator
        A = np.random.rand(512, 513) * 20.
        opA = lambda x : np.dot(A, x)
        opAadj = lambda x : np.dot(A.T, x)
        x = np.random.rand(513, 1)
        y = np.random.rand(512, 1)
        q1 = np.dot(np.dot(A, x).ravel(), y.ravel())
        q2 = np.dot(x.ravel(), np.dot(A.T, y).ravel())
        diff = abs(q1 - q2)
        diff_py = operators.misc.check_adjoint(opA, opAadj, x.shape, y.shape)
        self.assertTrue(abs(diff-diff_py) < 1e-5, msg="check_adjoint 2D failed (%e vs %e)" % (diff, diff_py))


    def test_div_grad(self):
        """
        Test if gradient and -divergence are adjoint of eachother
        """
        grad = operators.image.gradient
        minusdiv = lambda x : -operators.image.div(x)
        r = operators.misc.check_adjoint(grad, minusdiv, (512, 513), (2, 512, 513))
        self.assertTrue(abs(r) < 1e-10, msg="gradient and -divergence do not seem to be adjoint")


    def test_power_method(self):
        """
        Test the power method
        """
        # Create a random operator
        A = np.random.rand(512, 513) * 50.
        opA = lambda x : np.dot(A, x)
        opAadj = lambda x : np.dot(A.T, x)

        # "true" maximum singular value
        _, s, _ = np.linalg.svd(A)
        smax = s.max()

        # power method result
        x0 = np.random.rand(512)
        r = operators.misc.power_method(opA, opAadj, x0, n_it=15)
        self.assertTrue(abs(r - smax) < 1e-10, msg="power method failed on a random matrix (512, 513) with seed=0 (sigma_max = %e when it should be %e)" % (r, smax))


    def tearDown(self):
        pass






#~ if __name__ == "__main__":
    #~ unittest.main()


def test_suite_all_operators():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestOperators("test_check_adjoint"))
    testSuite.addTest(TestOperators("test_div_grad"))
    testSuite.addTest(TestOperators("test_power_method"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_operators()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)











