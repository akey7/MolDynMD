import numpy as np


# There is a blank string at the begining so the 1 position is H
element_symbols = ["", "H"]

class MolDynMDStretch:
    """
    This just computes the stretch energy of two atoms.
    The atoms are in 3 dimensional cartesian space.
    """
    

    def __init__(self, atom_types, atoms_xyz, atom_bonds):
        """
        Parameters
        ----------
        atom_types : list
            List of strings, each of which is the atomic number (such as 1 for 
            H) of the taom for that position.

        atom_positions : np.array
            Array of floats that are the positions of the atoms in angstroms.
            Shape (N, 3) where N is the number of atoms.

        atom_bonds : np.array
            Adjacency matrix of bond characteristic lengths (l_IJ_0),
            constants (k_IJ). Shape (N, N, 3). (i, i, 0) is l_IJ_0, 
            (i, i, 1) is k_IJ.
        """
        self._atoms_xyz = atoms_xyz
        self._atom_types = atom_types
        self._atom_bonds = atom_bonds
        self.timestep_integer = 0

    def __str__(self):
        """
        Outputs a long string of positions of atoms seperated by newlines
        """
        result = []

        result.append(str(len(self._atom_types)))
        result.append("")

        for i in range(len(self._atom_types)):
            atom_type = element_symbols[self._atom_types[i]]
            atom_xyz = self._atoms_xyz[i]
            result.append(f"{atom_type}\t{atom_xyz[0]}\t{atom_xyz[1]}\t{atom_xyz[2]}")

        return "\n".join(result)

    def timestep(self):
        """
        Steps one timestep of the model.
        """
        bonds = self._atom_bonds
        xyz = self._atoms_xyz
        types = self._atom_types

        for i in range(xyz.shape[0]):
            for j in range(xyz.shape[0]):
                if i !=j and bonds[i, j, 0] != 0:
                    type_i = types[i]
                    type_j = types[j]
                    l_IJ = bonds[i, j, 0]
                    k_IJ = bonds[i, j, 1]
                    r_i = xyz[i]
                    r_j = xyz[j]
                    print(f"Found bond from {r_i} to {r_j} type_i={type_i} type_j={type_j} l_IJ={l_IJ} k_IJ={k_IJ}")

        self.timestep_integer += 1


if __name__ == '__main__':
    atom_types = [1, 1]
    atom_xyz = np.array([[0, 0, 0], [0.74, 0, 0]])
    atom_bonds = np.array([
        [[0, 0], [1, 1]],
        [[1, 1], [0, 0]]
    ])
    mol_dyn = MolDynMDStretch(atom_types=atom_types, atoms_xyz=atom_xyz, atom_bonds=atom_bonds)
    
    mol_dyn.timestep()

    fn = f"xyz/trajectory_{mol_dyn.timestep_integer}.xyz"
    with open(fn, "a") as f:
        f.write(str(mol_dyn))
