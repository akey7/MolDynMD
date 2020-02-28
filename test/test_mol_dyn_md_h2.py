import numpy as np
from numpy import linalg

import pytest

from mol_dyn_md import MolDynMD


reference_length_of_HCl_m = 127.45e-12
force_constant = -1.0


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
    return md.graph, h, cl, md


def test_HCl_atom_count(hcl):
    graph, _, _, _ = hcl
    assert len(graph.nodes) == 2


def test_HCl_atom_types(hcl):
    graph, _, _, _ = hcl
    assert graph.nodes[0]["symbol"] == "H" and graph.nodes[1]["symbol"] == "Cl"


def test_HCl_reference_length(hcl):
    graph, _, _, _ = hcl
    bond = graph.edges[0, 1]
    assert bond["l_IJ_0"] == reference_length_of_HCl_m


def test_HCl_force_constant(hcl):
    graph, _, _, _ = hcl
    bond = graph.edges[0, 1]
    assert bond["k_IJ"] == force_constant


def test_HCl_positions(hcl):
    graph, _, _, _ = hcl
    expected_h_position = np.array([reference_length_of_HCl_m, 0., 0.])
    expected_cl_position = np.array([0., 0., 0.])
    actual_h_position = graph.nodes[0]["position"]
    actual_cl_position = graph.nodes[1]["position"]
    assert np.all(expected_cl_position == actual_cl_position) and np.all(expected_h_position == actual_h_position)


def test_HCl_bond(hcl):
    graph, h, cl, _ = hcl
    h_neighbors = list(graph[h])
    cl_neighbors = list(graph[cl])
    assert h_neighbors[0] == cl
    assert cl_neighbors[0] == h


def test_distances(hcl):
    """
    There is only one edge between h and cl. That edge's distance should be
    what was calculated when the edge was established.
    """
    graph, h, cl, _ = hcl
    for _, _, l_ij in graph.edges.data("l_ij"):
        assert l_ij == reference_length_of_HCl_m


def test_stretch_gradient(hcl):
    """
    Once again, because this is a diatomic, all the gradients should be the same.
    Also, there must be one timestep.
    """
    graph, _, _, md = hcl
    md.timestep()
    for _, _, v_stretch_gradient in graph.edges.data("v_stretch_gradient"):
        assert v_stretch_gradient == 0


def test_unit_vector(hcl):
    _, _, _, md = hcl
    r_j = np.array([3, 3, 3])
    r_i = np.array([0, 0, 0])
    expected = np.array([0.57735027, 0.57735027, 0.57735027])
    actual = md.unit_vector(r_i, r_j)
    assert np.allclose(expected, actual)


##########################################################################
# REGRESSION TESTS
##########################################################################

def test_velocities(hcl):
    graph, _, _, md = hcl
    md.timestep()
    actual_velocities = graph.nodes.data("velocity")
    expected_velocities = [np.array([-0.00029876, 0., 0.]), np.array([8.49312805e-06, 0.00000000e+00, 0.00000000e+00])]

    # actual_velocities is comprised of strange tuples that must be unpacked first.
    for (_, actual), expected in zip(actual_velocities, expected_velocities):
        assert np.allclose(actual, expected)


def test_positions(hcl):
    graph, _, _, md = hcl

    # The initial velocities are 0, so the positions after the first step can be zero
    # Once there is some acceleration in there, the atoms can move, so go for two
    # timesteps.
    md.timestep()
    md.timestep()

    actual_positions = graph.nodes.data("position")
    expected_positions = [np.array([1.2745e-10, 0.0000e+00, 0.0000e+00]),
                          np.array([8.49312805e-21, 0.00000000e+00, 0.00000000e+00])]

    for (_, actual), expected in zip(actual_positions, expected_positions):
        assert np.allclose(actual, expected)
