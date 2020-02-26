import numpy as np

import pytest

from mol_dyn_md import MolDynMD


reference_length_of_HCl_m = 127.45e-12
force_constant = -1.0


@pytest.fixture()
def hcl():
    md = MolDynMD()
    h_initial_position = np.array([reference_length_of_HCl_m, 0, 0])
    cl_initial_position = np.array([0, 0, 0])
    h_initial_velocity = np.array([0, 0, 0])
    cl_initial_velocity = np.array([0, 0, 0])
    h = md.add_atom("H", initial_position=h_initial_position, initial_velocity=h_initial_velocity)
    cl = md.add_atom("Cl", initial_position=cl_initial_position, initial_velocity=cl_initial_velocity)
    md.add_bond(h, cl, l_IJ_0=reference_length_of_HCl_m, k_IJ=force_constant)
    return md.graph, h, cl


def test_HCl_atom_count(hcl):
    graph, _, _ = hcl
    assert len(graph.nodes) == 2


def test_HCl_atom_types(hcl):
    graph, _, _ = hcl
    assert graph.nodes[0]["symbol"] == "H" and graph.nodes[1]["symbol"] == "Cl"


def test_HCl_reference_length(hcl):
    graph, _, _ = hcl
    bond = graph.edges[0, 1]
    assert bond["l_IJ_0"] == reference_length_of_HCl_m


def test_HCl_force_constant(hcl):
    graph, _, _ = hcl
    bond = graph.edges[0, 1]
    assert bond["k_IJ"] == force_constant


def test_HCl_positions(hcl):
    graph, _, _ = hcl
    expected_h_position = np.array([reference_length_of_HCl_m, 0, 0])
    expected_cl_position = np.zeros(3)
    actual_h_position = graph.nodes[0]["position"]
    actual_cl_position = graph.nodes[1]["position"]
    assert np.all(expected_cl_position == actual_cl_position) and np.all(expected_h_position == actual_h_position)

def test_HCl_bond(hcl):
    graph, h, cl = hcl
    h_neighbors = list(graph[h])
    cl_neighbors = list(graph[cl])
    assert h_neighbors[0] == cl
    assert cl_neighbors[0] == h
