import numpy as np


# There is a blank string at the begining so the 1 position is H
element_symbols = ["", "H"]

class MolDynMDStretch:
    """
    This just computes the stretch energy of two atoms.
    The atoms are in 3 dimensional cartesian space.
    """
    

    def __init__(self, *, atom_types, atom_positions, atom_velocities, atom_bonds):
        """
        Parameters
        ----------
        atom_types : list
            List of strings, each of which is the atomic number (such as 1 for 
            H) of the taom for that position.

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
        """
        self.atom_positions = atom_positions
        self.atom_types = atom_types
        self.atom_bonds = atom_bonds
        self.atom_velocities = atom_velocities
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
                    type_i = types[i]
                    type_j = types[j]
                    l_IJ_0 = bonds[i, j, 0]
                    k_IJ = bonds[i, j, 1]
                    r_i = xyz[i]
                    r_j = xyz[j]

                    v_str_ij = self.v_stretch_ij(r_i=r_i, r_j=r_j, k_IJ=k_IJ, l_IJ_0=l_IJ_0)

                    print(f"Found bond from {r_i} to {r_j} type_i={type_i} type_j={type_j} l_IJ_0={l_IJ_0} k_IJ={k_IJ} v_str_ij={v_str_ij}")

        self.timestep_integer += 1

    def v_stretch_ij(self, *, r_i, r_j, k_IJ, l_IJ_0):
        """
        Calculates the stretch energy of two bonded atoms.

        For more information on the parameters, please see:
        Quantum Chemistry, 7th ed, Ira N. Levine. pp. 636-637

        Parameters
        ----------
        r_i : np.array
            3 dimension row vector location of the location of the first atom.

        r_j : np.array
            3 dimension row vector location of the location of the second atom.

        k_IJ : float
            The force constant of the stretch.

        l_IJ_0 : float
            The reference length of the bond

        Returns
        -------
        float
            v_str_ij: The stretch energy between atoms i and j.
        """
        l2_norm = np.sqrt(np.sum(np.square(r_j - r_i)))
        v_str_ij = 1 / 2 * k_IJ * np.square(l2_norm - l_IJ_0)
        return v_str_ij


if __name__ == '__main__':
    atom_types = [1, 1]
    atom_xyz = np.array([[0, 0, 0], [1e-12, 0, 0]])
    atom_velocities = np.array([[0.01e-12, 0, 0], [-0.01e-12, 0, 0]])
    atom_bonds = np.array([
        [[0, 0], [0.74e-12, 1]],
        [[0.74e-12, 1], [0, 0]]
    ])

    mol_dyn = MolDynMDStretch(atom_types=atom_types,
                              atom_velocities=atom_velocities,
                              atom_positions=atom_xyz,
                              atom_bonds=atom_bonds)
    
    mol_dyn.timestep()

    fn = f"xyz/trajectory_{mol_dyn.timestep_integer}.xyz"
    with open(fn, "a") as f:
        f.write(str(mol_dyn))
