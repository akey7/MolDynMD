import numpy as np

import pytest

from mol_dyn_md import MolDynMD


@pytest.fixture()
def h2_no_velocity_fixture():
    md = MolDynMD()
    h1_initial_position = np.array([-32e-12, 0, 0])
    h2_initial_position = np.array([32e-12, 0, 0])
    h1_initial_velocity = np.array([0, 0, 0])
    h2_initial_velocity = np.array([0, 0, 0])
    md.add_atom("H", initial_position=h1_initial_position, initial_velocity=h1_initial_velocity)
    md.add_atom("H", initial_position=h2_initial_position, initial_velocity=h2_initial_velocity)
    return md


def test_h2_positions(h2_no_velocity_fixture):
    expected_positions = np.array([[-32e-12, 0, 0], [32e-12, 0, 0]])
    actual_positions = h2_no_velocity_fixture.positions
    print(actual_positions)
    assert True


