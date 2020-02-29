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
from numpy.linalg import norm


kg_per_amu = 1.66054e-27

atom_masses = {
    "H": 1,
    "C": 12,
    "O": 16,
    "Cl": 35,
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

    def __init__(self, timesteps=1000, dt=1.e-15, h=1e-15):
        self.graph = nx.Graph()
        self.atom_counter = 0
        self.frames = []
        self.dt = dt
        self.h = h
        self.timesteps = timesteps
        self.t = 0

    def add_atom(self, symbol, initial_position, initial_velocity):
        m = atom_masses[symbol]

        x = np.zeros((self.timesteps, 3))
        v = np.zeros((self.timesteps, 3))
        a = np.zeros((self.timesteps, 3))
        f = np.zeros((self.timesteps, 3))

        x[0] = initial_position
        v[0] = initial_velocity

        self.graph.add_node(self.atom_counter,
                            symbol=symbol,
                            x=initial_position,
                            v=initial_velocity,
                            m=m,
                            a=a,
                            f=f)

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
        atoms = self.graph.nodes

        if atom1 > self.atom_counter - 1:
            raise ValueError(f"atom1 {atom1} is out of range")

        if atom2 > self.atom_counter - 1:
            raise ValueError(f"atom2 {atom2} is out of range")

        if l_IJ_0 <= 0:
            raise ValueError(f"l_IJ_0 of {l_IJ_0} must be greater than or equal to 0")

        if k_IJ >= 0:
            raise ValueError(f"k_IJ of {k_IJ} should be negative")

        self.graph.add_edge(atom1, atom2, l_IJ_0=l_IJ_0, k_IJ=k_IJ)
