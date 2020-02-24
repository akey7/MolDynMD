"""
Notation in the file uses the style from Quantum Chemistry, 7th Ed by
Ira Levine, pp. 636-637. Note that there are some good unit conversions
in here.

However, in Introduction to Computational Chemistry by Jensen, pg. 64
has some interesting values for force fields also. Its explanation
of the stretch energy on pg. 25 Eqn. 2.3 is enlightening.
"""


import os

import networkx as nx
import numpy as np
import pandas as pd

from collections import namedtuple

# Defined outside the class so that it could be used by other code.
# However, do not modify it!

kg_per_amu = 1.66054e-27

atom_masses = {
    "H": 1.00784 * kg_per_amu,
    "Cl": 35.453 * kg_per_amu
}


class MolDynMD:
    """
    This is a molecular dynamics class. It only supports stretch energy

    Use MKS units for everything here and life will be easier.

    The nodes on the graph store the indecies into which the atom types
    can be found, as well as the positions and velocities (the latter two
    being store in NumPy arrays)

    The edges on the graph store the stretch energy parameters for the bond.
    """

    def __init__(self):
        """
        This initializes instance attributes which are accessible with
        properties.

        self.graph: Holds the 1,2 bonds in the model.

        self.symbols: The element symbols of each atom in a list

        self.positions: A numpy array with each row being a
            cartesian position of that atom.

        self.velocities: A numpy array with each row being the velocity
            of the atom in that positions.

        Note that the index of each atom position matches in the symbols
        list and positions and velocities arrays.
        """
        self.graph = nx.Graph()
        self.atom_counter = 0

    def add_atom(self, symbol, initial_position, initial_velocity):
        """
        Parameters
        ----------
        symbol : str
            The element symbol (H, He, C, Be, etc) of the atom being
            added.

        initial_position: np.array
            The starting position of the atom as a 3 element numpy array

        initial_velocity: np.array
            The initial velocity of the atom as a 3 element numpy array

        Returns
        -------
        int
            The index of the atom just created.

        Raises
        ------
        ValueError
            Raises a value error if the element symbol is unknown, or if the
            initial position or velocity arrays do not have exactly three
            elements.
        """
        if symbol not in atom_masses:
            raise ValueError(f"{symbol} is not a supported element symbol")

        if initial_position.shape[0] != 3:
            raise ValueError(f"Initial position of {initial_position} does not have 3 elements")

        if initial_velocity.shape[0] != 3:
            raise ValueError(f"Initial velocity of {initial_velocity} is not a 3 element array.")

        mass_kg = atom_masses[symbol]

        self.graph.add_node(self.atom_counter,
                            symbol=symbol,
                            position=initial_position,
                            velocity=initial_velocity,
                            mass_kg=mass_kg)

        self.atom_counter += 1

        return self.atom_counter - 1

    def add_bond(self, atom1, atom2, l_IJ_0, k_IJ):
        """
        This adds a bond as an edge on the graph.

        This is an undirected graph, so we don't need to add edges going in
        both directions.

        Parameters
        ----------
        atom1 : int
            Index of the first atom in the bond.

        atom2 : int
            Index of the second atom in the bond

        l_IJ_0 : float
            Reference length of the bond. See the Levine citation above.

        k_IJ : float
            Force constant of the bond.

        Raises
        ------
        ValueError
            Raised if the force constant >= 0, or if the reference length is
            negative. Also raised if atom1 or atom2 point to non existent atoms
        """
        if atom1 > self.atom_counter - 1:
            raise ValueError(f"atom1 {atom1} is out of range")

        if atom2 > self.atom_counter - 1:
            raise ValueError(f"atom2 {atom2} is out of range")

        if l_IJ_0 <= 0:
            raise ValueError(f"l_IJ_0 of {l_IJ_0 } must be greater than or equal to 0")

        if k_IJ >= 0:
            raise ValueError(f"k_IJ of {k_IJ} should be negative")

        self.graph.add_edge(atom1, atom2, l_IJ_0=l_IJ_0, k_IJ=k_IJ)

    def timestep(self):
        """
        This steps the simulations by the timestep that has been previously
        specified.

        Right now it does nothing.
        """
        pass

    def xyz_atom_list(self):
        """
        This returns a list of dictionaries appropriate to writing as a .xyz file.

        This method assumes that picometers have been used for distances. However, it
        can be useful to convert these to angstroms.

        Returns
        -------
        list[dict]
            The symbols and locations of the atoms.
        """
        rows = []

        for i in range(len(self.graph.nodes)):
            atom = self.graph.nodes[i]
            row = {
                "symbol": atom["symbol"],
                "x": atom["position"][0] * 1e10,
                "y": atom["position"][1] * 1e10,
                "z": atom["position"][2] * 1e10
            }
            rows.append(row)

        return rows


def main():
    """
    Though not strictly necessary, this avoids variables being planed in th
    outer scope of the module thereby creating shadowing problems.
    """
    md = MolDynMD()

    # Setup the initial and bond conditions
    reference_length_of_HCl_m = 127.45e-12
    force_constant = -1.0
    h_initial_position = np.array([reference_length_of_HCl_m, 0, 0])
    cl_initial_position = np.array([0., 0., 0.])
    h_initial_velocity = np.array([0., 0., 0.])
    cl_initial_velocity = np.array([0., 0., 0.])
    h1 = md.add_atom("H", initial_position=h_initial_position, initial_velocity=h_initial_velocity)
    cl = md.add_atom("Cl", initial_position=cl_initial_position, initial_velocity=cl_initial_velocity)
    md.add_bond(h1, cl, l_IJ_0=reference_length_of_HCl_m, k_IJ=force_constant)

    # Now time step it for a certain number of times
    number_of_timesteps = 100
    for i in range(number_of_timesteps):
        filename = os.path.join("xyz", f"trajectory_{i}.xyz")
        md.timestep()
        rows = md.xyz_atom_list()
        print(f"Writing time step {i} to {filename}")
        with open(filename, "w") as f:
            f.write(f"{len(rows)}\n")
            for row in rows:
                f.write(f"{row['symbol']}\t{row['x']}\t{row['y']}\t{row['z']}")

    # for row in rows:
    #     print(f"{row['symbol']}\t{row['x']}\t{row['y']}\t{row['z']}")


if __name__ == "__main__":
    main()
