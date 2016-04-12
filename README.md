[![Build Status](https://travis-ci.org/pierrepaleo/portal.svg?branch=master)](https://travis-ci.org/pierrepaleo/portal/)
[![Documentation](https://readthedocs.org/projects/portal/badge/?version=latest)](http://portal.readthedocs.org/en/latest/)

## Presentation


**Portal** (PythOn Reconstruction and Tomography ALgorithms) is a python package implementing various reconstruction techniques in signal processing.
It is primarily designed for being easy to use and extensible. The typical use of this package consists in two steps :

- Defining a problem
- Defining a way to solve this problem

A *problem* can be characterized by an *operator*. In signal processing, for example, linear operators can describe many common applications : denoising, deblurring, super-resolution, inpainting, tomographic reconstruction.
Portal comes with various operators (and their adjoint) for modeling these problems.

Once the problem settled, the next step is to choose a *reconstruction algorithm* to solve it.
In many cases, especially in Compressive Sensing frameworks, convex optimization algorithms are chosen. Portal implements some of the most used : FISTA, Chambolle-Pock, Conjugate (sub)gradient (more to come !).

## Installation

### Installation from the Python wheel

A Python wheel is provided for an easy installation. Simply download the wheel (.whl file) and install it with pip :

```bash
pip install --user wheel.whl
```

where ``wheel.whl`` is the wheel of the current version.

If you are *updating* PORTAL, you have to force the re-installation :

```bash
pip install --user --force-reinstall wheel.whl
```


### Installation from the sources


Alternatively, you can build and install this package from the sources.

```bash
git clone git://github.com/pierrepaleo/portal
```

* To generate a wheel, go in PORTAL root folder :

```bash
python setup.py bdist_wheel
```

The generated wheel can be installed with the aforementioned instructions.

* You can also directly install from the sources with

```bash
python setup.py install --user
```


### Dependencies

To use PORTAL, you must have Python > 2.7 and numpy >= 1.8. These should come with standard Linux distributions.

Numpy is the only component absolutely required for PORTAL. For special applications, the following are required :

   * The [ASTRA toolbox](https://github.com/astra-toolbox/astra-toolbox/) for tomography applications

   * ``pywt`` for Wavelets applications. This is a python module which can be installed with ``apt-get install python-pywt``

   * ``scipy.ndimage`` is used for convolutions with small kernel sizes. If not installed, all the convolutions are done in the Fourier domain, which can be slow.


**Note** : Python 3.* has not been tested yet.

## Documentation

A documentation is available in the ``doc/`` folder.


## Disclaimer

This module is at an early development stage. This means that the API may change at any time.




<!---

## Portal for tomographic reconstruction
---

### The `tomography` operator

The `tomography` operator in Portal relies on the [ASTRA toolbox](http://github.com/astra-toolbox), which should be installed beforehand.
Portal provides a simple wrapper for parallel 2D geometry, though ASTRA can handle many more geometries.
For now, the supported parameters are the width (pixels) of the slice, the number of projection angles, the rotation center, and the detector/slice super-sampling.

A simple example of Filtered Backprojection with Portal looks like this :

```
import portal

sino = portal.utils.io.edf_read('sino_0125.edf')
n_angles, n_px = sino.shape
rot_center = 1039.
tomo = portal.operators.tomography.AstraToolbox(n_px, n_angle, rot_center=rot_center)
rec_fbp = tomo.backproj(sino, filt=True)

portal.utils.io.edf_write('rec_0125.edf')
```
<br>

### Iterative techniques

Of course, Portal is more interesting when it comes to iterative techniques.
The user has first to decide which regularization type he will be using (TV, Wavelets, Tikhonov, ...). Then, an appropriate optimization algorithm should be chosen.
Optimization algorithms are designed to be versatile : here they take the `tomography` operator as a parameter, but they can handle other operators (problems) like blur for deconvolution.

An example of tomographic reconstruction with TV regularization, solved with the Chambolle-Pock algorithm, looks like this :


```python
import portal

sino = portal.utils.io.edf_read('sino_0125.edf')
n_angles, n_px = sino.shape
tomo = portal.operators.tomography.AstraToolbox(n_px, n_angle)

# Regularization parameter
Lambda_tv = 2.5
# Number of iterations
n_it = 500
# Define the tomographic operator
K = lambda x : tomo.proj(x)
# Define its adjoint
Kadj = lambda y : tomo.backproj(x, filt=False)

# Run the reconstruction algorithm
rec_tv = portal.algorithms.chambollepock.chambolle_pock_tv(sino, n_it, K, Kadj, Lambda_tv)
```
<br>
An example of tomographic reconstruction with Wavelets regularization, solved with the FISTA algorithm, looks like this :

```python
import portal

sino = portal.utils.io.edf_read('sino_0125.edf')
n_angles, n_px = sino.shape
tomo = portal.operators.tomography.AstraToolbox(n_px, n_angle)

w = portal.operators.wavelets.WaveletCoeffs()...

# Regularization parameter
Lambda_tv = 2.5
# Number of iterations
n_it = 500
# Define the wavelet-tomographic operator
K = lambda x : tomo.proj(x)
# Define its adjoint
Kadj = lambda y : tomo.backproj(x, filt=False)

# Run the reconstruction algorithm
rec_tv = portal.algorithms.chambollepock.chambolle_pock_tv(sino, n_it, K, Kadj, Lambda_tv)
```
<br>


--->

