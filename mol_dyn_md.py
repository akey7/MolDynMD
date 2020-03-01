import os
from dataclasses import dataclass
import networkx as nx
import numpy as np
from numpy.linalg import norm


kg_per_amu = 1.66054e-27

atom_masses = {
    "H": 1 * kg_per_amu,
    "C": 12 * kg_per_amu,
    "O": 16 * kg_per_amu,
    "Cl": 35 * kg_per_amu
}


@dataclass
class Atom:
    symbol: str
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

    Also assume that all atoms are indexed by i and j. j is the index of
    the "other" atom. j exerts forces on i.
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
        """
        Parameters
        ----------
        symbol: str
            The element symbol
            
        initial_position: np.array
            The 3 dimensional vector of the initial position
            
        initial_velocity: np.array
            The 3 dimensional vector
        """
        m = atom_masses[symbol]
        x = np.zeros((self.timesteps, 3))
        v = np.zeros((self.timesteps, 3))
        a = np.zeros((self.timesteps, 3))
        f = np.zeros((self.timesteps, 3))
        atom = Atom(symbol=symbol, m=m, x=x, v=v, a=a, f=f)
        x[0] = initial_position
        v[0] = initial_velocity
        self.graph.add_node(self.atom_counter, atom=atom)
        self.atom_counter += 1
        return self.atom_counter - 1

    def add_bond(self, atom_i, atom_j, l_IJ_0, k_IJ):
        """
        See Bond dataclass docstring above for citation of what these variables mean.
        
        Parameters
        ----------
        atom_i: int
            The integer ID of the graph node that is atom a
        """
        bond = Bond(l_IJ_0=l_IJ_0, k_IJ=k_IJ)
        self.graph.add_edge(atom_i, atom_j, bond=bond)

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

    def run(self, verbose=True):
        """
        This runs the trajectory over all the timesteps.

        Parameters
        ----------
        verbose: bool
            True to log each timestep after it is computed.
        """
        for t in range(self.timesteps - 1):
            if verbose:
                print(f"Timestep {t}")
            self.t = t
            self.iterate_timestep()

    def iterate_timestep(self):
        """
        Runs through one iteration of the timestep

        1. Calculate energy gradients
        2. Calculate forces
        3. Calculate accelerations
        4. Numeric integration

        This is meant to be used with the run() method. It relies on that method
        updating self.t so that all the timestamp indecies are correct.
        """
        t = self.t
        dt = self.dt

        # Iterate over each node, and find its edges for 1, 2 bonds
        # to compute stretch energies.
        for i, atom_i in self.graph.nodes(data="atom"):

            # Compute stretch forces from bonded atoms.
            for _, j, bond in self.graph.edges(nbunch=i, data="bond"):
                atom_j = self.graph.nodes[j]["atom"]
                grad = self.stretch_gradient(xi=atom_i.x, xj=atom_j.x, l_IJ_0=bond.l_IJ_0, k_IJ=bond.k_IJ)
                atom_i.f[t] += -grad * self.unit(xi=atom_i.x[t], xj=atom_j.x[t])

            # Now compute the acceleration on the atom i from its sum of forces
            atom_i.a[t] = atom_i.f[t] / atom_i.m

        # Now update the positions and velocities based on the accelerations
        for _, atom_i in self.graph.nodes(data="atom"):
            new_v = atom_i.v[t] + atom_i.a[t] * dt
            new_x = atom_i.x[t] + new_v * dt
            atom_i.v[t + 1] = new_v
            atom_i.x[t + 1] = new_x

    def trajectory_to_xyz_frames(self):
        """
        This writes the trajectory to a .xyz file

        Returns
        -------
        str
            A string, appropriate to write to a file, that contains
            .xyz format output that would animate an entire trajectory
        """
        scaling_factor = 1e10

        frames = []

        for t in range(self.timesteps):
            frames.append(f"{len(self.graph.nodes)}")
            frames.append(f"frame\t{t}\txyz")
            for _, atom in self.graph.nodes(data="atom"):
                position = atom.x[t]
                x = position[0] * scaling_factor
                y = position[1] * scaling_factor
                z = position[2] * scaling_factor
                frames.append(f"{atom.symbol}\t{x}\t{y}\t{z}")

        return frames
