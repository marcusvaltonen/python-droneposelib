# droneposelib

[![Build Status](https://travis-ci.com/marcusvaltonen/python-droneposelib.svg?branch=main)](https://travis-ci.com/marcusvaltonen/python-droneposelib)
![PyPI](https://img.shields.io/pypi/v/droneposelib)
![GitHub](https://img.shields.io/github/license/marcusvaltonen/python-droneposelib)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/marcusvaltonen/python-droneposelib.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/marcusvaltonen/python-droneposelib/context:python)
[![codecov](https://codecov.io/gh/marcusvaltonen/python-droneposelib/branch/main/graph/badge.svg)](https://codecov.io/gh/marcusvaltonen/python-droneposelib)

Python library for Visual-Inertial Odometry (VIO).
Wrapps the C++/Eigen library [DronePoseLib](https://github.com/marcusvaltonen/DronePoseLib).

## Solvers available
The current list of solvers are the following:

| Solver  | Approx. runtime\* | Max. solutions | Comment |
| --- | :---: | :---: | --- |
| `fEf` | 2.5 us | 4 | Valtonen Örnhag et al. (ArXiV 2021) |
| `frEfr` | 23\*\* us | 8 | Valtonen Örnhag et al. (ArXiV 2021) |
| `rEr` | 2.2 us | Valtonen Örnhag et al. (ArXiV 2021)\*\*\* |
\* Measured on a laptop with an Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz
\*\* If using the `use_fast_solver` option
\*\*\* Undocumented

## Installation
A pre-alpha release is available at PyPi, and can be installed using
```console
    $ pip install droneposelib
```
You may also compile the source code, see "Development".

## Examples
See the `example` directory for examples on how to use the solvers.

## Development
You are more than welcome to contribute your our other relevant solvers. More info soon.

## References
The code is related to the ArXiV paper [[link](https://arxiv.org/abs/2103.08286)]:

```
@misc{valtonenornhag-etal-2021-arxiv,
      title={Trust Your IMU: Consequences of Ignoring the IMU Drift},
      author={Marcus {Valtonen~Örnhag} and Patrik Persson and Mårten Wadenbäck and Kalle Åström and Anders Heyden},
      year={2021},
      eprint={2103.08286},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Please cite the paper if you are using the code for (academic) publications.
