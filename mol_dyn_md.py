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
import numpy.linalg as linalg


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

    def __init__(self, dt_s=1.e-15, grad_h_m=1e-15):
        """
        Private attributes

        self._graph: Holds the 1,2 bonds in the model.

        self._atom_counter: Holds a unique serial number for each atom.

        Note that the index of each atom position matches in the symbols
        list and positions and velocities arrays.

        Parameters
        ----------
        dt_s : float
            The timestep in seconds. Should probably be set on the order
            of femtoseconds.

        grad_h_m : float
            The h in the finite difference approximation of the gradients
            of the energy functions.
        """
        self.graph = nx.Graph()
        self.atom_counter = 0
        self.frames = []
        self.dt_s = dt_s
        self.grad_h_m = grad_h_m

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
                            mass_kg=mass_kg,
                            force_sum=np.array([0., 0., 0.]))

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

        r1 = self.graph.nodes[atom1]["position"]
        r2 = self.graph.nodes[atom2]["position"]
        l_ij = linalg.norm(r2 - r1)
        self.graph.add_edge(atom1, atom2, l_IJ_0=l_IJ_0, k_IJ=k_IJ, l_ij=l_ij, v_stretch_gradient=0)

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

    def timestep(self):
        """
        This steps one timestep in the MD. It calls methods for the following steps

        1. Compute the stretch energies.
        2. Compute the gradients of the energies. Steps 1 and 2 are in the same
            method call.
        3. Compute the forces
        4. Compute the accelerations
        5. Update velocities and positions with this data.
        """
        # Get all the nodes in the graph. The nodes are the atoms.
        atoms = self.graph.nodes

        # Forces are accumulated into vectors on each atom/node.
        # Reset them to zero here
        for i in self.graph:
            atoms[i]["force_sum"] = np.array([0., 0., 0.])

        # Calculate stretch forces
        for i in self.graph:
            position_i = atoms[i]["position"]
            for _, j, bond in self.graph.edges(nbunch=i, data=True):
                position_j = atoms[j]["position"]
                l_ij = linalg.norm(position_j - position_i)
                k_IJ = bond["k_IJ"]
                l_IJ_0 = bond["l_IJ_0"]
                grad = self.v_stretch_gradient(l_ij=l_ij, k_IJ=k_IJ, l_IJ_0=l_IJ_0)
                atoms[i]["force_sum"] += -grad * self.unit_vector(position_i, position_j)

        # Now compute the new velocities and positions
        for i in self.graph:
            atoms[i]["position"] += atoms[i]["velocity"] * self.dt_s
            accel = atoms[i]["force_sum"] / atoms[i]["mass_kg"]
            atoms[i]["velocity"] += accel * self.dt_s

    def unit_vector(self, r_i, r_j):
        """
        Computes the unit vector between two positions for calculating a force

        Parameters
        ----------
        r_i: np.array
            The first position.

        r_j: np.array
            The second position.

        Returns
        -------
        np.array
            The unit vector
        """
        difference = r_j - r_i
        norm = linalg.norm(difference)
        unit = difference / norm
        return unit

    def v_stretch_gradient(self, l_ij, k_IJ, l_IJ_0):
        """
        Given an edge, computes the gradient of the stretch energy
        between the associated with that edge.

        Parameters
        ----------
        l_ij: float
            The length of the bond.

        k_IJ: float
            The force constant of the bond.

        l_IJ_0: float
            The characteristic length of the bond

        Returns
        -------
        float
            The gradient!
        """
        h = self.grad_h_m

        v_ij = 0.5 * k_IJ * (l_ij - l_IJ_0) ** 2
        v_ij_plus_h = 0.5 * k_IJ * (l_ij + h - l_IJ_0) ** 2
        gradient = (v_ij_plus_h - v_ij) / h

        return gradient

    def run_trajectory(self, timesteps):
        """
        Runs the MD through a number of timesteps and returns a list of lists
        of dictionaries.

        Since the outcome of this will go into an animation, each list of atoms
        in a timestep will be called a frame.

        Parameters
        ----------
        timesteps: int
            The number of timesteps to run. The total amount of time in the run
            is timesteps * self.dt_s
        """
        self.frames = []

        for _ in range(timesteps):
            self.timestep()
            self.frames.append(self.xyz_atom_list())

    def write_trajectory_to_xyz_file(self, filename):
        """
        This writes the trajectory to a .xyz file

        Parameters
        ----------
        filename: str
            The absolute pathname.
        """
        fn = os.path.join("xyz", "trajectory.xyz")
        with open(fn, "w") as f:
            for idx, frame in enumerate(self.frames):
                print(len(frame), file=f)
                print(f"frame\t{idx}\txyz", file=f)
                for atom in frame:
                    print(f"{atom['symbol']}\t{atom['x']}\t{atom['y']}\t{atom['z']}", file=f)
