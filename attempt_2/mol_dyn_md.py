"""
Notation in the file uses the style from Quantum Chemistry, 7th Ed by
Ira Levine, pp. 636-637. Note that there are some good unit conversions
in here.

However, in Introduction to Computational Chemistry by Jensen, pg. 64
has some interesting values for force fields also. Its explanation
of the stretch energy on pg. 25 Eqn. 2.3 is enlightening.
"""


import networkx as nx
import numpy as np

from collections import namedtuple

# Defined outside the class so that it could be used by other code.
# However, do not modify it!
atom_masses = {
    "H": 1.00784 * 1.66054e-27
}


SymbolPositionVelocity = namedtuple("SymbolPositionVelocity", ("symbol", "position", "velocity"))


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
        self.symbols = []
        self.positions = None
        self.velocities = None

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

        index_for_new_atom = len(self.symbols)

        self.symbols.append(symbol)

        if self.positions is None:
            self.positions = initial_position.reshape(1, -1)
        else:
            self.positions = np.append(self.positions, initial_position).reshape(-1, 3)

        if self.velocities is None:
            self.velocities = initial_velocity.reshape(1, -1)
        else:
            self.velocities = np.append(self.velocities, initial_velocity).reshape(-1, 3)

        self.graph.add_node(index_for_new_atom)

        return index_for_new_atom

    @property
    def symbols_positions_velocities(self):
        result = [SymbolPositionVelocity(s, p, v) for s, p, v in zip(self.symbols, self.positions, self.velocities)]
        return result

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
        if atom1 > len(self.symbols) - 1:
            raise ValueError(f"atom1 {atom1} is out of range")

        if atom2 > len(self.symbols) - 1:
            raise ValueError(f"atom2 {atom2} is out of range")

        if l_IJ_0 <= 0:
            raise ValueError(f"l_IJ_0 of {l_IJ_0 } must be greater than or equal to 0")

        if k_IJ >= 0:
            raise ValueError(f"k_IJ of {k_IJ} should be negative")

        self.graph.add_edge(atom1, atom2, l_IJ_0=l_IJ_0, k_IJ=k_IJ)
