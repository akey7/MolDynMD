import numpy as np

import pytest

from mol_dyn_md import MolDynMD


@pytest.fixture()
def h2_md_initial_state():
    md = MolDynMD()
    h1_initial_position = np.array([-32e-12, 0, 0])
    h2_initial_position = np.array([32e-12, 0, 0])
    h1_initial_velocity = np.array([0, 0, 0])
    h2_initial_velocity = np.array([0, 0, 0])
    md.add_atom("H", initial_position=h1_initial_position, initial_velocity=h1_initial_velocity)
    md.add_atom("H", initial_position=h2_initial_position, initial_velocity=h2_initial_velocity)
    return md


def test_h2_positions(h2_md_initial_state):
    expected_positions = np.array([[-32e-12, 0, 0], [32e-12, 0, 0]])
    actual_positions = h2_md_initial_state.positions
    assert np.all(expected_positions == actual_positions)


def test_h2_velocities(h2_md_initial_state):
    expected_positions = np.array([[0, 0, 0], [0, 0, 0]])
    actual_positions = h2_md_initial_state.velocities
    assert np.all(expected_positions == actual_positions)


def test_h2_nodes(h2_md_initial_state):
    expected_nodes = {0, 1}
    actual_nodes = set(h2_md_initial_state.graph.nodes())
    assert expected_nodes == actual_nodes
