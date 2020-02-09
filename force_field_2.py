import numpy as np


# There is a blank string at the begining so the 1 position is H
element_symbols = ["", "H"]

class MolDynMDStretch:
    """
    This just computes the stretch energy of two atoms.
    The atoms are in 3 dimensional cartesian space.
    """
    

    def __init__(self, atom_types, atoms_xyz):
        """
        Parameters
        ----------
        atom_types : list
            List of strings, each of which is the atomic number (such as 1 for 
            H) of the taom for that position.

        atom_positions : np.array
            Array of floats that are the positions of the atoms in angstroms.
        """
        self._atoms_xyz = atoms_xyz
        self._atom_types = atom_types
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


if __name__ == '__main__':
    atom_types = [1, 1]
    atom_positions = np.array([[0, 0, 0], [0.74, 0, 0]])
    mol_dyn = MolDynMDStretch(atom_types=atom_types, atoms_xyz=atom_positions)
    
    fn = f"xyz/trajectory_{mol_dyn.timestep_integer}.xyz"
    with open(fn, "a") as f:
        f.write(str(mol_dyn))
