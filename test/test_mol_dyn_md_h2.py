from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm
import networkx as nx

import pytest
from pytest import approx

from mol_dyn_md import MolDynMD


reference_length_of_HCl_m = 127.45e-12
force_constant = -1.0


@dataclass
class HClFixture:
    md: MolDynMD
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
    md.add_bond(h, cl, l_IJ_0=reference_length_of_HCl_m, k_IJ=force_constant)
    return HClFixture(graph=md.graph, h=h, cl=cl, md=md)


def test_hcl_atom_count(hcl):
    assert len(hcl.graph.nodes) == 2


def test_hcl_bond_count(hcl):
    assert len(hcl.graph.edges) == 1


def test_stretch_gradient(hcl):
    """
    Test the analytical solution to the stretch gradient with a finite difference
    approximation.

    For the analytical solution, the answer should be 0. For the fd, it is a small
    float close to 0. That is why approx() is used, and why the approximate of 1.1
    is being used to make a search that will include 0.
    """

    h = 1e-5
    l_IJ_0 = 1
    k_IJ = -1
    xi = np.array([-0.5, 0., 0.])
    xj = np.array([0.5, 0., 0.])
    l_ij = norm(xj - xi)

    e = 0.5 * k_IJ * (l_ij - l_IJ_0) ** 2
    e_plus_h = 0.5 * k_IJ * (l_ij - l_IJ_0 + h) ** 2
    grad_fd = (e_plus_h - e) / 2

    grad_analytical = hcl.md.stretch_gradient(xi=xi, xj=xj, l_IJ_0=l_IJ_0, k_IJ=k_IJ)

    assert approx(grad_fd, 1.1) == grad_analytical


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
