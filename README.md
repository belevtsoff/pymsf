PyMSF
=====

A python library containig various routines for computing MSF, Master Stability Function (Pecora & Carroll, 1998).
The library is implemented in Cython for speed and seamless python integration.

Currently implemented tools allow:
* model integration (euler, rk2, rk4)
* extracting one limit-cycle period after it settled
* computing Floquet numbers of the MSF along a trajectory (only Euler integration atm)

Deps:
* numpy
* Cython >= 0.18

To build the modules, run:

    python2 setup.py build_ext --inplace

To run scripts, add the repo to your $PYTHONPATH. 

Currently, there are no docs and only a couple of docstrings, so you're on your own, good luck!
