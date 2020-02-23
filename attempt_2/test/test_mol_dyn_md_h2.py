import numpy as np

import pytest

from mol_dyn_md import MolDynMD, SymbolPositionVelocity


@pytest.fixture()
def h2_md_initial_state():
    md = MolDynMD()
    h1_initial_position = np.array([-32e-12, 0, 0])
    h2_initial_position = np.array([32e-12, 0, 0])
    h1_initial_velocity = np.array([0, 0, 0])
    h2_initial_velocity = np.array([0, 0, 0])
    h1 = md.add_atom("H", initial_position=h1_initial_position, initial_velocity=h1_initial_velocity)
    h2 = md.add_atom("H", initial_position=h2_initial_position, initial_velocity=h2_initial_velocity)
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


def test_symbol_positions_velocities(h2_md_initial_state):
    expected_symbol_positions_velocities = [
        SymbolPositionVelocity("H", np.array([[-32e-12, 0, 0]]), np.array([0, 0, 0])),
        SymbolPositionVelocity("H", np.array([[32e-12, 0, 0]]), np.array([0, 0, 0]))
    ]
    actual_symbol_positions_velocities = h2_md_initial_state.symbols_positions_velocities

    for (s1, p1, v1), (s2, p2, v2) in zip(expected_symbol_positions_velocities, actual_symbol_positions_velocities):
        assert s1 == s2 and np.all(p1 == p2) and np.all(v1 == v2)
