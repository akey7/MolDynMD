"""
Notation in the file uses the style from Quantum Chemistry, 7th Ed by
Ira Levine, pp. 636-637. Note that there are some good unit conversions
in here.

However, in Introduction to Computational Chemistry by Jensen, pg. 64
has some interesting values for force fields also. Its explanation
of the stretch energy on pg. 25 Eqn. 2.3 is enlightening.
"""


from dataclasses import dataclass
import networkx as nx
import numpy as np


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
    See Jensen pg. 25
    """
    k_ab: float
    r_ab_0: float


class MolDynMD:
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
        atom = Atom(m=m, x=x, v=v, a=a, f=f)
        x[0] = initial_position
        v[0] = initial_velocity
        self.graph.add_node(self.atom_counter, atom=atom)
        self.atom_counter += 1
        return self.atom_counter - 1

    def add_bond(self, atom_a, atom_b, r_ab_0, k_ab):
        """
        See Bond dataclass docstring above for citation of what these variables mean.
        """
        atoms = self.graph.nodes
        bond = Bond(r_ab_0=r_ab_0, k_ab=k_ab)
        self.graph.add_edge(atom_a, atom_b, bond=bond)
