import json
from pathlib import Path

import numpy as np
from ase.io import read
from flare.struc import Structure
from flare.utils.element_coder import NumpyEncoder


def extract_info(struc):
    f = np.array([x.forces for x in struc])
    e = np.array([x.energy for x in struc])

    return f, e


def reshape_forces(Y):
    # reshape training forces if needed
    if Y is not None:
        if len(Y.shape) != 1 or Y.shape[-1] != 3:
            # Y must have shape Nenv*3
            try:
                Y = Y.reshape((np.prod(Y.shape[:-1]) * 3))
            except TypeError:
                Y_ = []
                for a in Y:
                    Y_.extend(a)
                Y = np.array(Y_)
                Y = Y.reshape((np.prod(Y.shape[:-1]) * 3))

    return Y


def xyz_to_traj(infile, outfile=None,
                force_name='forces', energy_name='energy'):
    """
    Transform from xyz data format to the .json data
    format used in FLARE,
    which is used throughout the Raffy
    package for handling atomic environments.

    The FLARE package can be found here:
    https://github.com/mir-group/flare
    """

    if outfile is None:
        infile_path = Path(infile)
        outfile = str(infile_path.parent / str(infile_path.stem + ".json"))

    ase_traj = read(infile, index=':')
    for atoms in ase_traj:
        if not atoms.cell:
            atoms.set_cell([[200, 0, 0], [0, 200, 0], [0, 0, 200]])
            atoms.set_pbc([False, False, False])

    idx = np.arange(len(ase_traj))
    trajectory = []
    for i in idx:
        # forces are not imported with this method!
        struct = Structure.from_ase_atoms(ase_traj[i])
        try:
            struct.forces = ase_traj[i].arrays[force_name]
        except:
            pass
        try:
            struct.energy = ase_traj[i].info[energy_name]
        except:
            pass
        trajectory.append(struct.as_dict())

    with open(outfile, 'w') as fp:
        fp.write(
            '\n'.join(json.dumps(trajectory[i], cls=NumpyEncoder
                                 ) for i in np.arange(len(trajectory))))

    return trajectory


def load_structures(filename, force_name='forces', energy_name='energy'):
    """
    """
    filename_path = Path(filename)
    if filename_path.suffix == ".json":
        structures = Structure.from_file(filename)
    elif filename_path.suffix == ".xyz":
        _ = xyz_to_traj(filename, str(
            filename_path.parent) + "flare_structures.json",
            force_name=force_name, energy_name=energy_name)
        structures = Structure.from_file(str(
            filename_path.parent) + "flare_structures.json")
    else:
        print("""Data format not recognized.
Supported formats are .xyz and FLARE .json format""")

    return structures
