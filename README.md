# Raffy
Ridge-regression Atomistic Force Fields in PYthon

Use this package to train and validate force fields for single- and multi-element materials.
The force fields are trained using ridge regression on local atomic environment descritptors computed in the "Atomic Cluster Expansion" (ACE) framework.
The package can also be used to classify local atomic environments according to their ACE local descriptor.

To install the package, clone the repository and then pip install:

    git clone https://github.com/ClaudioZeni/Raffy
    cd Raffy
    pip install .

The installation process should take 1 to 5 minutes on a standard laptop.

Two notebooks are available in the examples folder, one showcases the training and validation of a linear potential for Si, the other demonstrates how to use the Raffy package to classify local atomic environments on a sample MD trajectory of a Au nanoparticle.

The package uses ASE to handle .xyz files, MIR-FLARE to handle local atomic environments, NUMPY and SCIPY for fast computation, and RAY for multiprocessing.

The package has been tested on Ubuntu 20.04
