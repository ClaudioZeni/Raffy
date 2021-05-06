# Raffy
Ridge-regression Atomistic Force Fields in PYthon

Use this package to train and validate force fields for single- and multi-element materials.
The force fields are trained using ridge regression on local atomic environment descritptors computed in the "Atomic Cluster Expansion" framework.

To install the package, clone the repository and run the following command from the main folder:

    pip install .

The package uses ASE to handle .xyz files, FLARE to handle local atomic environments, NUMPY and SCIPY for fast computation, and RAY for multiprocessing.

