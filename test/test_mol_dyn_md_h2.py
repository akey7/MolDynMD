from dataclasses import dataclass
import numpy as np
import networkx as nx

import pytest

from mol_dyn_md import MolDynMD


reference_length_of_HCl_m = 127.45e-12
force_constant = -1.0


@dataclass
class HClFixture:
    graph: nx.Graph
    h: int
    cl: int


@pytest.fixture()
def hcl():
    md = MolDynMD()
    h_initial_position = np.array([reference_length_of_HCl_m, 0., 0.])
    cl_initial_position = np.array([0., 0., 0.])
    h_initial_velocity = np.array([0., 0., 0.])
    cl_initial_velocity = np.array([0., 0., 0.])
    h = md.add_atom("H", initial_position=h_initial_position, initial_velocity=h_initial_velocity)
    cl = md.add_atom("Cl", initial_position=cl_initial_position, initial_velocity=cl_initial_velocity)
    md.add_bond(h, cl, r_ab_0=reference_length_of_HCl_m, k_ab=force_constant)
    return HClFixture(graph=md.graph, h=h, cl=cl)


def test_hcl_atom_count(hcl):
    assert len(hcl.graph.nodes) == 2


def test_hcl_bond_count(hcl):
    assert len(hcl.graph.edges) == 1


# def test_positions(hcl):
#     graph, _, _, md = hcl
#
#     # The initial velocities are 0, so the positions after the first step can be zero
#     # Once there is some acceleration in there, the atoms can move, so go for two
#     # timesteps.
#     md.timestep()
#     md.timestep()
#
#     actual_positions = graph.nodes.data("position")
#     expected_positions = [np.array([1.2745e-10, 0.0000e+00, 0.0000e+00]),
#                           np.array([8.49312805e-21, 0.00000000e+00, 0.00000000e+00])]
#
#     for (_, actual), expected in zip(actual_positions, expected_positions):
#         assert np.allclose(actual, expected)
