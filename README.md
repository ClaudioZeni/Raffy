# Raffy
Ridge-regression Atomistic Force Fields in PYthon

Use this package to train and validate force fields for single- and multi-element materials.
The force fields are trained using ridge regression on local atomic environment descritptors computed in the "Atomic Cluster Expansion" (ACE) framework.
The package can also be used to classify local atomic environments according to their ACE local descriptor.


## Installation
To install the package, clone the repository and then pip install:

    git clone https://github.com/ClaudioZeni/Raffy
    cd Raffy
    pip install .

The installation process should take 1 to 5 minutes on a standard laptop.


## Examples
Two notebooks are available in the examples folder, one showcases the training and validation of a linear potential for Si, the other demonstrates how to use the Raffy package to classify local atomic environments on a sample MD trajectory of a Au nanoparticle.


## Dependancies
The package uses ASE to handle[a link](https://pypi.org/project/ase/) .xyz files, MIR-FLARE[a link](https://github.com/mir-group/flare) to handle local atomic environments, NUMPY[a link](https://numpy.org/) and SCIPY[a link](https://www.scipy.org/) for fast computation, and RAY[a link](https://ray.io/) for multiprocessing.

The package has been tested on Ubuntu 20.04.


## References
If you use RAFFY in your research, or any part of this repository, please cite the following paper:

[1] Claudio Zeni, Kevin Rossi, Aldo Glielmo, and Stefano de Gironcoli, "Compact atomic descriptors enable accurate predictions via linear models", The Journal of Chemical Physics 154, 224112 (2021) https://doi.org/10.1063/5.0052961 


