import os
import numpy as np
from mol_dyn_md import MolDynMD

"""
All distances are in angstroms.

All times are in picoseconds
"""

md = MolDynMD(timesteps=1000, dt=1e-3)

co2_bond_length = 1.16
force_constant = -1

# First CO2 molecule
c_1_initial_position = np.array([0., 0., 0.])
c_1_initial_velocity = np.array([0., 0., 0.])

o1_initial_position = np.array([co2_bond_length, 0., 0.])
o1_initial_velocity = np.array([-0.01, 0., 0.])

o2_initial_position = np.array([-co2_bond_length, 0., 0.])
o2_initial_velocity = np.array([0.01, 0., 0.])

c1 = md.add_atom("C", initial_position=c_1_initial_position, initial_velocity=c_1_initial_velocity)
o1 = md.add_atom("O", initial_position=o1_initial_position, initial_velocity=o1_initial_velocity)
o2 = md.add_atom("O", initial_position=o2_initial_position, initial_velocity=o2_initial_velocity)

md.add_bond(c1, o1, l_IJ_0=co2_bond_length, k_IJ=force_constant)
md.add_bond(c1, o2, l_IJ_0=co2_bond_length, k_IJ=force_constant)

# Second CO2 molecule
c_2_initial_position = np.array([0., 0., co2_bond_length * 3])
c_2_initial_velocity = np.array([0., 0., 0.])

o3_initial_position = np.array([0., co2_bond_length, co2_bond_length * 3])
o3_initial_velocity = np.array([0., -0.01, 0.])

o4_initial_position = np.array([0., -co2_bond_length, co2_bond_length * 3])
o4_initial_velocity = np.array([0., 0.01, 0.])

c2 = md.add_atom("C", initial_position=c_2_initial_position, initial_velocity=c_2_initial_velocity)
o3 = md.add_atom("O", initial_position=o3_initial_position, initial_velocity=o3_initial_velocity)
o4 = md.add_atom("O", initial_position=o4_initial_position, initial_velocity=o4_initial_velocity)

md.add_bond(c2, o3, l_IJ_0=co2_bond_length, k_IJ=force_constant)
md.add_bond(c2, o4, l_IJ_0=co2_bond_length, k_IJ=force_constant)

md.run()

filename = os.path.join("xyz", "CO2 Trajectory.xyz")
with open(filename, "w") as f:
    frames = md.trajectory_to_xyz_frames(scaling_factor=1, step=250)
    print("\n".join(frames), file=f)

filename = os.path.join("xyz", "CO2 Tracjectory.csv")
df = md.tracjectory_to_dataframe()
df.to_csv(filename, index=False)