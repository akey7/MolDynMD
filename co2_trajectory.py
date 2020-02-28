import os
import numpy as np
from mol_dyn_md import MolDynMD


md = MolDynMD()

bond_length_m = 116.3e-12
force_constant = -1.0

o1_initial_position = np.array([bond_length_m, 0., 0.])
o2_initial_position = np.array([-bond_length_m, 0., 0.])
c_initial_position = np.array([0., 0., 0.])

o1_initial_velocity = np.array([0., 0., 0.])
o2_initial_velocity = np.array([0., 0., 0.])
c_initial_velocity = np.array([0., 0., 0.])

o1 = md.add_atom("O", initial_position=o1_initial_position, initial_velocity=o1_initial_velocity)
o2 = md.add_atom("O", initial_position=o2_initial_position, initial_velocity=o2_initial_velocity)
c = md.add_atom("C", initial_position=c_initial_position, initial_velocity=c_initial_velocity)
md.add_bond(o1, c, l_IJ_0=bond_length_m, k_IJ=force_constant)
md.add_bond(o2, c, l_IJ_0=bond_length_m, k_IJ=force_constant)

fn = os.path.join("xyz", "trajectory.xyz")
md.run_trajectory(1000)
md.write_trajectory_to_xyz_file(fn)
