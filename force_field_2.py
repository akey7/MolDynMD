import numpy as np
from numpy.linalg import norm


# There is a blank string at the begining so the 1 position is H
element_symbols = ["", "H"]

class MolDynMDStretch:
    """
    This just computes the stretch energy of two atoms.
    The atoms are in 3 dimensional cartesian space.
    """
    

    def __init__(self, *, atom_types, atom_masses, atom_positions, atom_velocities, atom_bonds, dt_s, grad_h_m):
        """
        To make your life easier, please use mks units. Thanks!

        Parameters
        ----------
        atom_types : list
            List of strings, each of which is the atomic number (such as 1 for 
            H) of the atom for that position.

        atom_masses : np.array
            Array of floats that are the masses of each atom in kg.
            Shape (N) where N is the number of atoms.

        atom_positions : np.array
            Array of floats that are the positions of the atoms in meters.
            Shape (N, 3) where N is the number of atoms. Each row is a 3
            dimension position vector.

        atom_velocities : np.array
            Array of floats that are the velocities of the atoms in m/s.
            Shape (N, 3) where N is the number of atoms. Each row is a
            3 dimension position vector.

        atom_bonds : np.array
            Adjacency matrix of bond characteristic lengths (l_IJ_0),
            constants (k_IJ). Shape (N, N, 3). (i, i, 0) is l_IJ_0, 
            (i, i, 1) is k_IJ. See the v_stretch_ij method below for
            more on these parameters and where to read more.

        dt_s : float
            Time step, in seconds, for the numeric integration step.

        grad_h_m : float
            For finite difference approximation, this is the delta used
            to compute the finite differences. In meters.
        """
        self.atom_positions = atom_positions
        self.atom_types = atom_types
        self.atom_bonds = atom_bonds
        self.atom_velocities = atom_velocities
        self.atom_masses = atom_masses
        self.dt_s = dt_s
        self.grad_h_m = grad_h_m
        self.timestep_integer = 0

    def __str__(self):
        """
        Outputs a long string of positions of atoms seperated by newlines
        """
        result = []

        result.append(str(len(self.atom_types)))
        result.append("")

        for i in range(len(self.atom_types)):
            atom_type = element_symbols[self.atom_types[i]]
            atom_xyz = self.atom_positions[i]
            result.append(f"{atom_type}\t{atom_xyz[0]}\t{atom_xyz[1]}\t{atom_xyz[2]}")

        return "\n".join(result)

    def timestep(self):
        """
        WARNING: THIS IS A FIRST PASS AT THIS MODEL. IT ASSUMES A DIATOMIC
        MOLECULE. SO I AM SKIPPING THE FORCE SUMMATION STEP FOR NOW.

        Steps one timestep of the model:

        1. Energies
        2. gradients
        3. Forces
        4. Accelerations
        5. Update velocities and positions
        """
        bonds = self.atom_bonds
        xyz = self.atom_positions
        types = self.atom_types

        for i in range(xyz.shape[0]):
            for j in range(xyz.shape[0]):
                if i !=j and bonds[i, j, 0] != 0:
                    l_IJ_0 = bonds[i, j, 0]
                    k_IJ = bonds[i, j, 1]
                    r_i = xyz[i]
                    r_j = xyz[j]

                    # Compute the stretch energy and its gradient
                    l_ij = norm(r_j - r_i)
                    v_str_ij, grad_str_ij = self.v_stretch_ij(l_ij=l_ij, k_IJ=k_IJ, l_IJ_0=l_IJ_0)

                    # Compute the unit vector from i to j
                    unit_ij = (r_j - r_i) / l_ij

                    # Compute force SEE NOTE ABOUT DIATOMIC ASSUMPTION ABOVE
                    f_ij = -grad_str_ij * unit_ij

                    print(f"Bond from {r_i} to {r_j} l_IJ_0={l_IJ_0} k_IJ={k_IJ} l_ij={l_ij} v_str_ij={v_str_ij} grad_str_ij={grad_str_ij} unit_ij={unit_ij} f_ij={f_ij}")

        self.timestep_integer += 1

    def v_stretch_ij(self, *, l_ij, k_IJ, l_IJ_0):
        """
        Calculates the stretch energy of two bonded atoms. Also calculates
        the gradient of that energy with a finite difference approximation.

        For more information on the parameters, please see:
        Quantum Chemistry, 7th ed, Ira N. Levine. pp. 636-637.

        For finite differences, see the derivation from the Taylor expansion at
        https://en.wikipedia.org/wiki/Finite_difference_method#Derivation_from_Taylor's_polynomial
        https://en.wikipedia.org/wiki/Difference_quotient#Overview

        Parameters
        ----------
        l_ij : float
            The distance between the two atoms in question.

        k_IJ : float
            The force constant of the stretch.

        l_IJ_0 : float
            The reference length of the bond

        Returns
        -------
        float, float
            First element v_str_ij: The stretch energy between atoms i and j.
            Second element: Gradient of the stretch energy.
        """
        def v_str_ij(x): return 1 / 2 * k_IJ * np.square(x - l_IJ_0)

        h = self.grad_h_m
        grad_str_ij = (v_str_ij(l_ij + h) - v_str_ij(l_ij)) / h

        return v_str_ij(l_ij), grad_str_ij


if __name__ == '__main__':
    atom_types = [1, 1]
    atom_xyz = np.array([[0, 0, 0], [1e-12, 0, 0]])
    atom_velocities = np.array([[0.01e-12, 0, 0], [-0.01e-12, 0, 0]])
    atom_bonds = np.array([
        [[0, 0], [0.74e-12, 1]],
        [[0.74e-12, 1], [0, 0]]
    ])
    atom_masses = np.array([1.634e-27, 1.634e-27])  # The masses of H atoms in kg

    mol_dyn = MolDynMDStretch(atom_types=atom_types,
                              atom_masses=atom_masses,
                              atom_velocities=atom_velocities,
                              atom_positions=atom_xyz,
                              atom_bonds=atom_bonds,
                              dt_s=0.1e-15,
                              grad_h_m=1e-15)
    
    mol_dyn.timestep()

    fn = f"xyz/trajectory_{mol_dyn.timestep_integer}.xyz"
    with open(fn, "a") as f:
        f.write(str(mol_dyn))
