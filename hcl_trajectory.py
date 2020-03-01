import os
import numpy as np
from mol_dyn_md import MolDynMD


md = MolDynMD(timesteps=1000)

reference_length_of_HCl_m = 1.2745e-10
force_constant = -0.01
h_initial_position = np.array([reference_length_of_HCl_m * 0.999, 0., 0.])
cl_initial_position = np.array([0., 0., 0.])
h_initial_velocity = np.array([0., 0., 0.])
cl_initial_velocity = np.array([0., 0., 0.])
h1 = md.add_atom("H", initial_position=h_initial_position, initial_velocity=h_initial_velocity)
cl = md.add_atom("Cl", initial_position=cl_initial_position, initial_velocity=cl_initial_velocity)
md.add_bond(h1, cl, l_IJ_0=reference_length_of_HCl_m, k_IJ=force_constant)

md.run()

filename = os.path.join("xyz", "HCl Trajectory.xyz")
with open(filename, "w") as f:
    frames = md.trajectory_to_xyz_frames()
    print("\n".join(frames), file=f)

filename = os.path.join("xyz", "HCL Tracjectory.csv")
df = md.tracjectory_to_dataframe()
df.to_csv(filename, index=False)
