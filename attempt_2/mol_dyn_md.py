import networkx as nx
import numpy as np

# Defined outside the class so that it could be used by other code.
# However, do not modify it!
atom_masses = {
    "H": 1.00784 * 1.66054e-27
}


class MolDynMD:
    """
    This is a molecular dynamics class. It only supports stretch energy

    Use MKS units for everything here and life will be easier.
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

        return index_for_new_atom