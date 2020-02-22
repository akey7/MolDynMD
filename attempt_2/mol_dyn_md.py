from collections import namedtuple

import networkx as nx

# Defined outside the class so that it could be used by other code.
# However, do not modify it!
atom_masses = {
    "H": 1.00784 * 1.66054e-27
}

class MolDynStretch:
    """
    This is a molecular dynamics class. It only supports stretch energy
    """

    def __init__(self):
        """
        This initializes instance attributes which are accessible with
        properties.

        self.graph: Holds the 1,2 bonds in the model.

        self.atom_serial: Hods an incrementing serial number
        """
        self.graph = nx.Graph()
        self.atom_serial = 0

    def add_atom(self, symbol, x_m, y_m, z_m, vx_m_per_s, v_y_m_per_s, v_z_m_per_s):
        """
        Parameters
        ----------
        symbol : str
            The element symbol (H, He, C, Be, etc) of the atom being
            added.

        Returns
        -------
        int
            The serial number of the atom being added. Starts at
            0.

        Raises
        ------
        ValueError
            Raises a value error if the element symbol is unknown.
        """
        if symbol not in atom_masses:
            raise ValueError(f"{symbol} is not a supported element symbol")


