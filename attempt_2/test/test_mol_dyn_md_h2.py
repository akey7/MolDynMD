import numpy as np

import pytest

from mol_dyn_md import MolDynMD


bond_length_of_H2_m = 74.0e-12
force_constant = -1.2


@pytest.fixture()
def h2_md_initial_state():
    """
    Set up an H2 molecule to do all the tests.
    """

    md = MolDynMD()
    h1_initial_position = np.array([-32e-12, 0, 0])
    h2_initial_position = np.array([32e-12, 0, 0])
    h1_initial_velocity = np.array([0, 0, 0])
    h2_initial_velocity = np.array([0, 0, 0])
    h1 = md.add_atom("H", initial_position=h1_initial_position, initial_velocity=h1_initial_velocity)
    h2 = md.add_atom("H", initial_position=h2_initial_position, initial_velocity=h2_initial_velocity)
    md.add_bond(h1, h2, l_IJ_0=bond_length_of_H2_m, k_IJ=force_constant)
    return md


def test_h2_atom_count(h2_md_initial_state):
    assert len(h2_md_initial_state.graph.nodes) == 2
