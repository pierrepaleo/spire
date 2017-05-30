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

#
# The following code was adapted from Matlab to Python
# (c) Ryota Tomioka - http://tomioka.dk
#

def svd(Op, Opadj, op_input_shape, op_output_shape, rank=None, niter=None, blocksize=None, float32=True):
    """
    constructs a nearly optimal rank "rank" approximation USV' to "A",
    using "niter" full iterations of a block Lanczos method
    of block size "blocksize".
    The method starts with a M*blocksize random matrix, where "A" is
    a M*N matrix.

    Parameters
    -----------
    rank: integer, optional (default is 6)
        Rank of the approximation USV' of the A (number of singular values)
    niter: integer, optional (default is 2)
        Number of iterations of the Lanczos method. Increasing it yields better
        accuracy.
    blocksize: integer, optional (default is rank+2)
        Block size of the block Lanczos iterations. Increasing it yields better
        accuracy.
    float32: bool
        Force the computation to be carried on float32 precision.

    Reference
    ---------
    Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
    Finding structure with randomness: Stochastic algorithms
    for constructing approximate matrix decompositions,
    arXiv:0909.4061 [math.NA; math.PR], 2009

    This code was adapted from Matlab
    http://ttic.uchicago.edu/~ryotat/softwares/tensor/preview/release/pca.html
    """

    if rank is None: rank = 6
    if niter is None: niter = 2
    if blocksize is None: blocksize = rank+2

    #if op_input_shape.ndim != 2:
    #    raise ValueError("Input matrix should be 2D")
    if rank > np.min(np.array(np.prod(op_input_shape), np.prod(op_output_shape))):
        raise ValueError("Rank should be less than the minimum of the matrix \
            dimensions")
    if rank > blocksize:
        raise ValueError("rank should be less than blocksize")


    dtype = np.float32 if float32 else np.float64

    n = np.prod(op_input_shape)
    m = np.prod(op_output_shape)

    # Build matrix H
    On = np.ones(op_output_shape, dtype=dtype)
    H = np.zeros((n, blocksize), dtype=dtype)
    for b in range(blocksize):
        R = np.random.rand(*op_output_shape) # TODO: scale ?
        H[:, b] =  Opadj(2*R - On).ravel()

    # Initialize F to its final size and fill its leftmost block with H.
    F = np.zeros((n, (niter + 1)*blocksize), dtype=dtype)
    F[:n, :blocksize] = H
    # Apply A'*A to H a total of its times,
    # augmenting F with the new H each time.
    for it in range(niter):
        for b in range(blocksize):
            H[:, b] = Opadj(Op(H[:, b].reshape(op_input_shape))).ravel()
            F[:n, it*blocksize+b] = H[:, b]

    # Form a matrix Q whose columns constitute an orthonormal basis
    # for the columns of F.
    Q, R = np.linalg.qr(F)

    # SVD A*Q to obtain approximations to the singular values
    # and left singular vectors of A; adjust the right singular vectors
    # of A*Q to approximate the right singular vectors of A.
    Q2 = np.zeros((m, Q.shape[-1]), dtype=dtype)
    for b in range(Q.shape[-1]):
        Q2[:, b] = Op((Q[:, b].reshape(op_input_shape))).ravel()

    U, S, V2 = np.linalg.svd(Q2, full_matrices=0)
    #~ V = Q.dot(V2) # <= This is an error in the Matlab implementation (or, I did not understand the syntax).
    # The following line is correct.
    V = V2.dot(Q.T)

    return U, S, V

