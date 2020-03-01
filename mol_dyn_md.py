from dataclasses import dataclass
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


@dataclass
class Atom:
    m: float
    x: np.array
    v: np.array
    f: np.array
    a: np.array


@dataclass
class Bond:
    """
    See Levine pg. 636
    """
    k_IJ: float
    l_IJ_0: float


class MolDynMD:
    """
    Velocities, positions, accelerations and forces are represneted as
    3 dimensional row vectors.

    Where these vectors are needed for summing (as is the case of forces)
    or where these vectors are used as trajectories, each new vector in
    the sequence is added as a row.
    """

    def __init__(self, timesteps=1000, dt=1.e-15, h=1e-15):
        """
        Set up the simulation run.

        Parameters
        ---------
        timesteps: int
            The number of timesteps to run in the simulation.

        dt: float
            The timestep. Should probably keep on the order of
            femtoseconds.

        h: float
            The step size to use for numeric differentiation.
        """
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
        atom = Atom(m=m, x=x, v=v, a=a, f=f)
        x[0] = initial_position
        v[0] = initial_velocity
        self.graph.add_node(self.atom_counter, atom=atom)
        self.atom_counter += 1
        return self.atom_counter - 1

    def add_bond(self, atom_a, atom_b, l_IJ_0, k_IJ):
        """
        See Bond dataclass docstring above for citation of what these variables mean.
        """
        bond = Bond(l_IJ_0=l_IJ_0, k_IJ=k_IJ)
        self.graph.add_edge(atom_a, atom_b, bond=bond)

    def stretch_gradient(self, xi, xj, l_IJ_0, k_IJ):
        """
        Calculate the stretch gradient between two positions and using the
        bond constants given.

        See Levine pg. 636 for an explanation of the variables l_IJ_0, k_IJ

        Parameters
        ----------
        xi: np.array
            The array of x, y, z coordinates for atom i

        xj: np.array
            The array of x, y, z coordinates for atom j

        l_IJ_0: float
            The characteristic length of the bond.

        Returns
        -------
        float
            The value of the stretch gradient.
        """
        l_ij = norm(xj - xi)
        grad = k_IJ * (2 * l_ij - 2 * l_IJ_0) / 2
        return grad

    def unit(self, xi, xj):
        """
        Returns the unit vector pointing from xi toward xj

        Parameters
        ----------
        xi: np.array
            The array of x, y, z coordinates for atom i

        xj: np.array
            The array of x, y, z coordinates for atom j

        Returns
        -------
        np.array
            The unit vector pointing from xi toward xj
        """
        return (xj - xi) / norm(xj - xi)

    def force(self, xi, xj, grad):
        """
        Calculate the forces exerted on xi by xj with the given gradient.

        Parameters
        ----------
        xi: np.array
            The array of x, y, z coordinates for atom i

        xj: np.array
            The array of x, y, z coordinates for atom j

        grad: float
            The gradient

        Returns
        -------
        np.array
            The force vector.
        """
        return -grad * self.unit(xi=xi, xj=xj)

    def acceleration(self, f, m):
        """
        Parameters
        ----------
        f: np.array
            A matrix of force vectors. Each individual force vector is a 3
            dimensional row vector. So to get the final sum of forces,
            the columns are summed.

        m: float
            The mass of the object upon which the force is acting.

        Returns
        -------
        np.array
            The 3 dimensional acceleration vector that results from force
            and the mass.
        """
        return f.sum(axis=0) / m
