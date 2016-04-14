[![Build Status](https://travis-ci.org/pierrepaleo/vita.svg?branch=master)](https://travis-ci.org/pierrepaleo/vita/)
[![Documentation](https://readthedocs.org/projects/vita/badge/?version=latest)](http://vita.readthedocs.org/en/latest/)

## Presentation


**VITA** (Various Image processing and Tomography Algorithms) is a python package providing
wrappers and utilities for tomography and image processing algorithms.

It is divided in several parts:

- The ``tomography`` submodule provides a simple wrapper of the [ASTRA toolbox](https://github.com/astra-toolbox/astra-toolbox/) for parallel beam geometry. It also provides sinogram pre-processing (denoising, rings artifacts correction) and dataset exploration utilities.
- The ``operators`` submodule implements various linear operators: FFT, Wavelet Transform, Convolution, tomographic projector, and other image operators.
- The ``algorithms`` submodule provides proximal algorithms (FISTA, Chambolle-Pock) for minimizing an objective function involving a linear operator.




## Installation

### Installation from a Python wheel

If a Python wheel is provided, you can install it with pip :

```bash
pip install --user wheel.whl
```

where ``wheel.whl`` is the wheel of the current version.

If you are *updating* VITA, you have to force the re-installation :

```bash
pip install --user --force-reinstall wheel.whl
```


### Installation from the sources


You can build and install this package from the sources.

```bash
git clone git://github.com/pierrepaleo/vita
```
* You can directly install from the sources with

```bash
python setup.py install --user
```

* Alternatively, you can generate a wheel and then install this wheel:

```bash
python setup.py bdist_wheel
pip install --user dist/wheel.whl
```


### Dependencies

To use VITA, you must have Python > 2.7 and numpy >= 1.8. These should come with standard Linux distributions.

Numpy is the only component absolutely required for PORTAL. For special applications, the following are required :

   * The [ASTRA toolbox](https://github.com/astra-toolbox/astra-toolbox/) for tomography applications

   * ``pywt`` for Wavelets applications. This is a python module which can be installed with ``apt-get install python-pywt``



**Note** : This module should be compatible with Python 3. However, all the dependencies must have a Python 3 version installed.


## Documentation

A documentation is available in the ``doc/`` folder.


## Disclaimer

This module is at an early development stage. This means that the API may change at any time.



