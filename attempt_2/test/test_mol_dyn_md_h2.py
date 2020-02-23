import numpy as np

import pytest

from mol_dyn_md import MolDynMD


reference_length_of_HCl_m = 127.45e-12
force_constant = -1.0


@pytest.fixture()
def hcl():
    """
    Set up an H2 molecule to do all the tests.
    """

    md = MolDynMD()
    h_initial_position = np.array([reference_length_of_HCl_m, 0, 0])
    cl_initial_position = np.array([0, 0, 0])
    h_initial_velocity = np.array([0, 0, 0])
    cl_initial_velocity = np.array([0, 0, 0])
    h1 = md.add_atom("H", initial_position=h_initial_position, initial_velocity=h_initial_velocity)
    cl = md.add_atom("Cl", initial_position=cl_initial_position, initial_velocity=cl_initial_velocity)
    md.add_bond(h1, cl, l_IJ_0=reference_length_of_HCl_m, k_IJ=force_constant)
    return md


def test_HCl_atom_count(hcl):
    assert len(hcl.graph.nodes) == 2


def test_HCl_atom_types(hcl):
    assert hcl.graph.nodes[0]["symbol"] == "H" and hcl.graph.nodes[1]["symbol"] == "Cl"


def test_HCl_reference_length(hcl):
    bond = hcl.graph.edges[0, 1]
    assert bond["l_IJ_0"] == reference_length_of_HCl_m
