import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from flare import struc
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.utils.element_coder import NumpyEncoder, Z_to_element
from matplotlib import pyplot as plt

#######################
### UNIT CONVERSION ###
#######################


def ry_to_ev(energy):
    """
    """
    return 13.6056980659 * energy


def bohr_to_a(length):
    """
    """
    return 0.529177208 * length


def ry_bohr_to_ev_a(force):
    """
    """
    return 25.71104309541616 * force


def ev_to_ry(energy):
    """
    """
    return energy / 13.6056980659


def a_to_bohr(length):
    """
    """
    return length / 0.529177208


def ev_a_to_ry_bohr(force):
    """
    """
    return force / 25.71104309541616


#########################
### FORMAT CONVERSION ###
#########################


def xyz_to_example(folder, xyz_filename):

    folder = Path(folder)
    xyz_file = str(folder / xyz_filename)
    json_file = str(folder / "flare_structures.json")
    example_folder = str(folder / "json/")

    if not os.path.isdir(example_folder):
        os.makedirs(example_folder)
    _ = xyz_to_traj(xyz_file, json_file)
    structures = load_structures(json_file)
    write_structures_to_example(structures, example_folder,
                                filenames=["%s.example" % (str(i)) for i in np.arange(len(structures))])


def xyz_to_traj(infile, outfile=None, force_name='forces', energy_name='energy'):
    """
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


def struct_to_example(struct, ev_ang =  True):
    """
    """
    example = {}
    if ev_ang:
        example['lattice_vectors'] = struct._cell.tolist()
        example['energy'] = [struct.energy, 'eV']
        example['atomic_position_unit'] = 'cartesian'
        example['unit_of_length'] = 'angstrom'
        forces = struct.forces.tolist()
        pos = struct.positions.tolist()

    else:
        example['lattice_vectors'] = a_to_bohr(struct._cell).tolist()
        en_ry = ev_to_ry(struct.energy)
        example['energy'] = [en_ry, 'Ry']
        example['atomic_position_unit'] = 'cartesian'
        example['unit_of_length'] = 'bohr'
        forces = ev_a_to_ry_bohr(struct.forces).tolist()
        pos = a_to_bohr(struct.positions).tolist()

    example['atoms'] = [[i, s, p, f] for i, s, p, f in zip(
        (1 + np.arange(len(struct))).tolist(), struct.species_labels, pos, forces)]
    return example


def write_structures_to_example(structures, filename_directory, filenames=None):
    """
    """
    if filenames:
        json_filenames = filenames
    else:
        json_filenames = os.listdir(filename_directory)

    for struct, filename in zip(structures, json_filenames):
        # try:
            example = struct_to_example(struct)
            with open(filename_directory + "/" + filename, 'w') as f:
                json.dump(example, f)
        # # except:
        #     print("Could not do %s" % (filename))
        #     pass


def dict_to_struc(data, single_atom_energy=None):
    """
    """
    positions = np.zeros((len(data['atoms']), 3))
    forces = np.zeros((len(data['atoms']), 3))
    species = []
    for i, atom in enumerate(data['atoms']):
        positions[i] = atom[2]
        forces[i] = atom[3]
        species.append(atom[1])
    energy = data['energy'][0]
    cell = np.asarray(data['lattice_vectors'])

    if data['unit_of_length'] == 'bohr':
        positions = bohr_to_a(positions)
        cell = bohr_to_a(cell)

    if data['energy'][1] == 'Ry':
        energy = ry_to_ev(energy)

    if data['unit_of_length'] == 'bohr' and data['energy'][1] == 'Ry':
        forces = ry_bohr_to_ev_a(forces)

    # Normalize energy by removing single-atom energy
    if single_atom_energy:
        energy = energy - len(data['atoms']) * single_atom_energy

    structure = struc.Structure(
        cell=cell, species=species, positions=positions, forces=forces, energy=energy)
    return structure


def save_as_xyz(structures, outfile="trajectory.xyz", energy=True, labels=False):
    """
    """
    if energy:
        for i in np.arange(len(structures)):
            if i == 0:
                to_xyz(structures[i], write_file=outfile, append=False,
                       dft_energy=structures[i].energy,
                       dft_forces=structures[i].forces, labels=labels)
            else:
                to_xyz(structures[i], write_file=outfile, append=True,
                       dft_energy=structures[i].energy,
                       dft_forces=structures[i].forces, labels=labels)
    else:
        for i in np.arange(len(structures)):
            if i == 0:
                to_xyz(structures[i], write_file=outfile, append=False,
                       dft_forces=structures[i].forces, labels=labels)
            else:
                to_xyz(structures[i], write_file=outfile, append=True,
                       dft_forces=structures[i].forces, labels=labels)


def save_as_json(structures, outfile="flare_structures.json"):
    """
    """
    flare_structures = {}
    for i in np.arange(len(structures)):
        flare_structures[i] = structures[i].as_dict()
    with open(outfile, 'w') as fp:
        fp.write(
            '\n'.join(json.dumps(flare_structures[i],
                                 cls=NumpyEncoder) for i in np.arange(len(flare_structures))))


def maml_to_structures(folder, input_filename, output_filename="flare_structures.json"):
    if type(folder) == str:
        folder = Path(folder)
    with open(folder / input_filename) as f:
        data = json.load(f)

    json_structures = []
    for struc in data:
        nat = len(struc['structure']['sites'])
        cell = np.array(struc['structure']['lattice']['matrix'])
        energy = struc['outputs']['energy']
        forces = np.array(struc['outputs']['forces'])
        positions = np.zeros((nat, 3))
        species = []
        for i, atom in enumerate(struc['structure']['sites']):
            positions[i] = atom['xyz']
            species.append(atom['label'])

        json_structures.append(Structure(
            cell=cell, species=species,
            positions=positions, forces=forces, energy=energy))
    save_as_json(json_structures, folder / output_filename)


def to_xyz(struct, extended_xyz: bool = True, print_stds: bool = False,
           print_forces: bool = False, print_max_stds: bool = False,
           print_energies: bool = False, predict_energy=None,
           dft_forces=None, dft_energy=None, timestep=-1,
           write_file: str = '', append: bool = False, labels=None) -> str:
    """
    Convenience function which turns a structure into an extended .xyz
    file; useful for further input into visualization programs like VESTA
    or Ovito. Can be saved to an output file via write_file.

    :param print_stds: Print the stds associated with the structure.
    :param print_forces:
    :param extended_xyz:
    :param print_max_stds:
    :param write_file:
    :return:
    """
    species_list = [Z_to_element(x) for x in struct.coded_species]
    xyz_str = ''
    xyz_str += f'{len(struct.coded_species)} \n'

    # Add header line with info about lattice and properties if extended
    #  xyz option is called.
    if extended_xyz:
        cell = struct.cell

        xyz_str += f'Lattice="{cell[0,0]} {cell[0,1]} {cell[0,2]}'
        xyz_str += f' {cell[1,0]} {cell[1,1]} {cell[1,2]}'
        xyz_str += f' {cell[2,0]} {cell[2,1]} {cell[2,2]}"'
        if timestep > 0:
            xyz_str += f' Timestep={timestep}'
        if predict_energy:
            xyz_str += f' PE={predict_energy}'
        if dft_energy is not None:
            xyz_str += f' DFT_PE={dft_energy}'
        xyz_str += f' Properties=species:S:1:pos:R:3'

        if print_stds:
            xyz_str += ':stds:R:3'
            stds = struct.stds
        if print_forces:
            xyz_str += ':forces:R:3'
            forces = struct.forces
        if print_max_stds:
            xyz_str += ':max_std:R:1'
            stds = struct.stds
        if labels:
            xyz_str += ':tags:R:1'
            clustering_labels = struct.local_energy_stds
        if print_energies:
            if struct.local_energies is None:
                print_energies = False
            else:
                xyz_str += ':local_energy:R:1'
                local_energies = struct.local_energies
        if dft_forces is not None:
            xyz_str += ':dft_forces:R:3'
        xyz_str += '\n'
    else:
        xyz_str += '\n'

    for i, pos in enumerate(struct.positions):
        # Write positions
        xyz_str += f"{species_list[i]} {pos[0]} {pos[1]} {pos[2]}"

        # If extended XYZ: Add in extra information
        if print_stds and extended_xyz:
            xyz_str += f" {stds[i,0]} {stds[i,1]} {stds[i,2]}"
        if print_forces and extended_xyz:
            xyz_str += f" {forces[i,0]} {forces[i,1]} {forces[i,2]}"
        if print_energies and extended_xyz:
            xyz_str += f" {local_energies[i]}"
        if print_max_stds and extended_xyz:
            xyz_str += f" {np.max(stds[i,:])} "
        if labels and extended_xyz:
            xyz_str += f" {clustering_labels[i]} "
        if dft_forces is not None:
            xyz_str += f' {dft_forces[i, 0]} {dft_forces[i,1]} ' \
                f'{dft_forces[i, 2]}'
        if i < (len(struct.positions) - 1):
            xyz_str += '\n'

    # Write to file, optionally
    if write_file:
        if append:
            fmt = 'a'
        else:
            fmt = 'w'
        with open(write_file, fmt) as f:
            f.write(xyz_str)
            f.write("\n")

    return xyz_str


def convert_example_to_struct(folder, single_atom_energy):
    """
    """
    structures = []
    for i, file in enumerate(os.listdir(folder)):
        with open(folder + "/" + file) as json_file:
            data = json.load(json_file)
            structures.append(dict_to_struc(data, single_atom_energy))
        if i % 1000 == 0:
            print(i)

    save_as_xyz(structures, outfile=folder + "/../" + "trajectory.xyz")
    save_as_json(structures, outfile=folder + "/../" + "flare_structures.json")

    # Check results:
    structures_from_json = Structure.from_file(
        folder + "/../" + "flare_structures.json")

    print(structures_from_json[-1] == structures[-1])

    return structures


#######################
### FLARE SHORTCUTS ###
#######################

def load_mgp(filename):
    """
    """
    from flare.mgp import MappedGaussianProcess
    with open(filename) as f:
        mgp_json = json.load(f)
    mgp = MappedGaussianProcess.from_dict(mgp_json)
    cutoffs = mgp_json['cutoffs']
    return mgp, cutoffs


def load_structures(filename):
    """
    """
    filename_path = Path(filename)
    if filename_path.suffix == ".json":
        structures = Structure.from_file(filename)
    elif filename_path.suffix == ".xyz":
        structures = xyz_to_traj(filename, str(
            filename_path.parent + "flare_structures.json"))

    return structures


def predict_on_structures(mgp, structures, cutoffs):
    """
    """
    data = {}

    for i in range(len(structures)):

        print(i)
        data[i] = {}
        forces = np.zeros((len(structures[i]), 3))
        stress = np.zeros((len(structures[i]), 6))
        var = np.zeros(len(structures[i]))
        tot_en = 0
        try:
            for j in range(len(structures[i])):
                forces[j], var[j], stress[j], e = mgp.predict(
                    AtomicEnvironment(structures[i], j, cutoffs))
                tot_en += e
        except:
            print("Could not do %i" % (i))
            pass

        data[i]['pred_forces'] = forces
        data[i]['pred_en'] = tot_en
        data[i]['pred_en_peratom'] = tot_en / len(structures[i])
        data[i]['forces'] = structures[i].forces
        data[i]['en'] = structures[i].energy
        data[i]['en_peratom'] = structures[i].energy / structures[i].nat
        data[i]['force_err'] = np.sum(
            (data[i]['pred_forces'] - data[i]['forces'])**2, axis=1)**0.5
        data[i]['en_peratom_err'] = data[i]['en_peratom'] - \
            data[i]['pred_en_peratom']
        data[i]['en_err'] = data[i]['en'] - data[i]['pred_en']

    return data


def plot_potential(mgp):
    pot_2b = mgp.maps['twobody'].as_dict()['maps'][0][0]
    start, end = mgp.maps['twobody'].as_dict()['bounds'][0]
    ax = plt.figure(figsize=(12, 8))
    plt.plot(np.linspace(start, end, len(pot_2b)), pot_2b)
    plt.xlabel(r"Distance [$\text{\AA}$]")
    plt.ylabel("Energy [eV]")
    return ax


def get_errors(mgp_filename, structure_filename):
    """
    """
    mgp, cutoffs = load_mgp(mgp_filename)
    structures = load_structures(structure_filename)

    data = predict_on_structures(mgp, structures, cutoffs)

    df = pd.DataFrame.from_dict(data).T
    return df


def train_gp(tr_indx, hyps, cutoffs, opt, trajectory, ker, cpus, energy=False):
    """
    """
    from flare.gp import GaussianProcess

    # Create GP object
    if ker == 'mb':
        hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'sigm', 'lsm', 'noise']
        kernels = ['2', '3', 'mb']
    else:
        hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']
        kernels = ['2', '3']

    gp_model = GaussianProcess(kernels=kernels, hyps=hyps, cutoffs=cutoffs,
                               hyp_labels=hyp_labels,
                               opt_algorithm='BFGS', parallel=True,
                               n_cpus=cpus,
                               maxiter=10,
                               )

    # Add points to training set
    for i in tr_indx:
        if energy:
            gp_model.update_db(
                trajectory[i], trajectory[i].forces, energy=trajectory[i].energy)
        else:
            gp_model.update_db(
                trajectory[i], trajectory[i].forces)

    if opt:
        # train a gp model WITH max marginal likelihood opt
        gp_model.train(logger_name="hyps_evolution_%i_%s" % (len(tr_indx), ker),
                       print_progress=True)
    else:
        # train a gp model WITHOUT max marginal likelihood opt
        gp_model.update_L_alpha()

    return gp_model


def test_gp(gp_model, trajectory, test_indx, cutoffs, energy):
    """
    """
    preds = []
    vars_ = []
    trues = []
    pred_en = []
    true_en = []
    for count, t in enumerate(test_indx):
        print(count)
        struct = trajectory[t]
        total_energy = 0
        for i in range(len(struct)):
            pred_ = np.zeros(3)
            var_ = np.zeros(3)
            true_ = np.zeros(3)

            for d in range(1, 4):
                pred = gp_model.predict(
                    AtomicEnvironment(struct, i, cutoffs), d)

                true_[d - 1] = struct.forces[i, d - 1]
                var_[d - 1] = pred[1]
                pred_[d - 1] = pred[0]
            if energy:
                total_energy += gp_model.predict_local_energy(
                    AtomicEnvironment(struct, i, cutoffs))

            preds.append(pred_)
            vars_.append(var_)
            trues.append(true_)

        if energy:
            pred_en.append(total_energy)
            true_en.append(struct.energy)

    preds = np.asarray(preds)
    trues = np.asarray(trues)
    vars_ = np.asarray(vars_)

    if energy:
        pred_en = np.asarray(pred_en)
        true_en = np.asarray(true_en)

    errors = np.sum((np.asarray(trues) - np.asarray(preds))**2, axis=1)**0.5

    if energy:
        energy_errors = np.abs(pred_en - true_en)
        return errors, trues, preds, energy_errors, true_en, pred_en

    else:
        return errors, trues, preds, None, None, None
