import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cholesky
from sklearn.decomposition import PCA

from . import compute_descriptors as cd
from . import utils as ut
from flare import struc


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class LinearPotential():

    def __init__(self, n_bodies=3, ns=4, ls=6, cutoff=None,
                 species=None, add_squares=False, basis='bessel'):
        self.n_bodies = n_bodies
        self.ns = ns
        self.ls = ls
        self.rc = cutoff
        self.species = np.array(species)
        self.species_dict = {key: value for (value, key) in enumerate(species)}
        self.n_species = len(species)
        self.add_squares = add_squares
        self.basis = basis

        if self.n_bodies == '2.5':
            self.g_func = cd.Descr25(self.rc, self.ns, self.ls, self.species,
                                     self.basis)
        elif self.n_bodies == '3':
            self.g_func = cd.Descr3(self.rc, self.ns, self.ls, self.species,
                                    self.basis)
        else:
            print("n_bodies not understood. use '2.5' or '3' ")

    def add_square_g(self, g, dg, X, compute_forces):
        # Generate squared descriptors
        g2 = g**2
        if compute_forces:
            j = 0
            dg2 = []
            for i, x in enumerate(X):
                dg2.extend(2*np.einsum('d, mcd -> mcd',
                           g[i], dg[j:j+x.nat]))
                j += x.nat
            dg2 = np.array(dg2)
        # Append squared descriptors
        g = np.append(g, g2, axis=-1)
        if compute_forces:
            dg = np.append(dg, dg2, axis=-1)

        return g, dg

    def adjust_g(self, g, dg, X, compute_forces=True, train_pca=False):
        dg_reshape = []
        if len(g.shape) == 3:
            g = np.array([x.sum(axis=0) for x in g])
        for i in np.arange(len(g)):
            if compute_forces and dg[i].shape == 4:
                dg_reshape.extend(np.einsum('ndmc -> mcd', dg[i]))

        if compute_forces:
            # Has shape M, 3, D
            if len(dg_reshape) > 0:
                dg = np.array(dg_reshape)

        if self.add_squares:
            g, dg = self.add_square_g(g, dg, X, compute_forces)
        if self.use_pca:
            if train_pca:
                pca = PCA(n_components=self.nc_pca)
                pca.fit(g)
                self.pca_rotation = pca.components_
            g = np.einsum('nd, id -> ni', g, self.pca_rotation)
            if compute_forces:
                dg = np.einsum('mcd, id -> mci', dg, self.pca_rotation)

        return g, dg

    def get_g(self, X, g=None, dg=None, compute_forces=False,
              ncores=1, train_pca=False):
        if (g is None or (dg is None and compute_forces)):
            g, dg = self.g_func.compute(X, compute_dgvect=compute_forces,
                                        ncores=ncores)

        g, dg = self.adjust_g(g, dg, X, compute_forces, train_pca)
        return g, dg

    def fit(self, X, Y=None, Y_en=None, noise=1e-8,
            g=None, dg=None, ncores=1, pca_comps=None,
            compute_forces=True):

        self.noise = noise
        if pca_comps is not None:
            self.use_pca = True
            self.nc_pca = pca_comps
        else:
            self.use_pca = False

        if (type(X) == list and type(X[0]) == struc.Structure):
            Y, Y_en = ut.extract_info(X)

        Y = ut.reshape_forces(Y)
        g, dg = self.get_g(X, g, dg, compute_forces, ncores,
                           train_pca=self.use_pca)

        if compute_forces:
            # Reshape to Nenv*3, D
            dg = np.reshape(dg, (dg.shape[0]*3, dg.shape[2]))

        # Cover both energy, force and force/energy training
        if Y_en is not None and compute_forces:
            g_tot = np.concatenate((-dg, g), axis=0)
            Y_tot = np.concatenate((Y, Y_en), axis=0)
        elif Y_en is not None and not compute_forces:
            g_tot = g
            Y_tot = Y_en
        else:
            g_tot = -dg
            Y_tot = Y

        # ftf shape is (S, S)
        gtg = np.einsum('na, nb -> ab', g_tot, g_tot)
        # Add regularization
        noise = self.noise*np.ones(len(gtg))
        noise[-len(X):] = self.noise
        gtg[np.diag_indices_from(gtg)] += noise
        # Cholesky Decomposition to find alpha
        L_ = cholesky(gtg, lower=True)
        # Calculate fY
        gY = np.einsum('na, n -> a', g_tot, Y_tot)
        # Find Alpha
        alpha = cho_solve((L_, True), gY)
        self.alpha = alpha

    def fit_local(self, X, Y, g, dg, noise=1e-8):
        self.noise = noise
        dg = np.reshape(dg, (dg.shape[0]*3, dg.shape[2]))
        Y = ut.reshape_forces(Y)
        g_tot = -dg
        Y_tot = Y
        # ftf shape is (S, S)
        gtg = np.einsum('na, nb -> ab', g_tot, g_tot)
        # Add regularization
        noise = self.noise*np.ones(len(gtg))
        noise[-len(X):] = self.noise
        gtg[np.diag_indices_from(gtg)] += noise
        # Cholesky Decomposition to find alpha
        L_ = cholesky(gtg, lower=True)
        # Calculate fY
        gY = np.einsum('na, n -> a', g_tot, Y_tot)
        # Find Alpha
        alpha = cho_solve((L_, True), gY)
        self.alpha = alpha

    def predict(self, structures, compute_forces=True,
                g=None, dg=None, ncores=1):

        G, dG = self.get_g(structures, g, dg, compute_forces, ncores)
        es, fs = [], []
        nat_counter = 0
        for i in np.arange(len(structures)):
            e = np.dot(G[i], self.alpha)
            es.append(e)
            if compute_forces:
                f = -np.einsum('mcd, d -> mc',
                               dG[nat_counter:nat_counter+structures[i].nat],
                               self.alpha)
                fs.extend(f)
                nat_counter += structures[i].nat
        return np.array(es), np.array(fs)

    def save(self, folder='.'):
        name = []
        name.append(self.n_bodies)
        name.extend(self.species)
        name = np.array(name, dtype='str')
        name = '_'.join(name)

        folder = Path(folder)
        alpha_filename = str(
            folder / str("alpha_potential_%s.npy" % (name)))
        param_filename = str(
            folder / str("potential_%s.json" % (name)))
        parameters = {
            'ns': self.ns,
            'ls': self.ls,
            'cutoff': self.rc,
            'nbodies': self.n_bodies,
            'species': list(self.species),
            'add_squares': (self.add_squares),
            'basis': self.basis,
            'alpha_filename': alpha_filename
        }
        np.save(alpha_filename, self.alpha)
        with open(param_filename, 'w') as f:
            json.dump(parameters, f, cls=NpEncoder)

        print("Potential saved at %s" % (param_filename))

    @staticmethod
    def load(filename):
        with open(filename) as f:
            parameters = json.load(f)

        new_potential = LinearPotential(
            parameters['nbodies'],
            parameters['ns'],
            parameters['ls'],
            parameters['cutoff'],
            parameters['species'],
            parameters['add_squares'],
            parameters['basis'])
        new_potential.alpha = np.load(
            parameters['alpha_filename'], allow_pickle=True)

        return new_potential
