class MolDynTwoAtom:
    """
    This is a class that implements a simple molecular dynamics
    algorithm

    A means Angstrom
    """

    def __init__(self,
                r0_A,
                epsilon,
                r1_A,
                r2_A, 
                v1_A_per_fs,
                v2_A_per_fs, 
                dt_fs, 
                grad_delta_A):
        """
        The instance attributes that are created by this method
        are:

        Parameters
        ----------
        r0_A: float
            The minimum energy distance of the Lennard Jones potential.

        epsilon:
            Depth of potential well. In kcal A^6 / mol.

        r1_A: float
            The position (in 1 dimension) of the first atom in
            the system. Initial position.

        r2_A: float
            The position (in 1 dimension) of the second atom in
            the system. Initial position.

        v1_A_per_fs: float
            The velocity (in 1 dimension) of the first atom
            in the system. Initial velocity.

        v2_A_per_fs: float
            The velocity (in 1 dimension) of the second atom
            of the system. Initial velocity

        dt_fs: float
            The time step for the numeric integration.

        grad_delta_A: float
            The delta to use in numeric differentiation.
            In A.
        """
        self.r0_A = r0_A
        self.epsilon = epsilon
        self.r1_A = r1_A 
        self.r2_A = r2_A
        self.v1_A_per_fs = v1_A_per_fs
        self.v2_A_per_fs = v2_A_per_fs
        self.dt_fs = dt_fs
        self.grad_delta_A = grad_delta_A
        self.time_fs = 0

        self.v1s = [v1_A_per_fs]
        self.v2s = [v2_A_per_fs]
        self.r1s = [r1_A]
        self.r2s = [r2_A]
        self.timesteps = [0]

    def timestep(self):
        """
        This updates positions and velocities for one time step.
        """

        # Compute the gradients
        grad_12, grad_21 = self.compute_gradients()
        f_12 = -grad_12
        f_21 = -grad_21

        # Foolishly assume masses are 1 and compute accelerations
        a_1 = f_12 / 1
        a_2 = f_21 / 1

        # Step the velocities
        self.v1_A_per_fs += a_1 * self.dt_fs
        self.v2_A_per_fs += a_2 * self.dt_fs

        # Step the positions
        self.r1_A += self.v1_A_per_fs * self.dt_fs
        self.r2_A += self.v2_A_per_fs * self.dt_fs

        # Store the position and velocity history
        self.v1s.append(self.v1_A_per_fs)
        self.v2s.append(self.v2_A_per_fs)
        self.r1s.append(self.r1_A)
        self.r2s.append(self.r2_A)

        # Store the timestep
        self.timesteps.append(self.time_fs)

        # Increment the time counter
        self.time_fs += self.dt_fs

    def compute_gradients(self):
        """
        This computes the gradients at each position.
        
        Returns
        -------
        float, float:
            The gradients for r1 and r2
        """
        r_12 = self.r1_A - self.r2_A
        r_21 = self.r2_A - self.r1_A
        delta = self.grad_delta_A
        
        grad_12 = (self.lj(r_12 + delta) - self.lj(r_12)) / delta
        grad_21 = (self.lj(r_21 + delta) - self.lj(r_21)) / delta
        
        return grad_12, grad_21

    def lj(self, r_A):
        """
        This calculates the Lennard Jones potential of a bond length.
        
        Parameters
        ----------
        r_A: float or np.array
            Length(s) of the bond(s) in A.
            
        Returns
        -------
        float
            Array of floats of potential energies. kcal/mol
        """
        repulsive = (self.r0_A / r_A) ** 12
        attractive = -2 * (self.r0_A / r_A) ** 6
        energy = self.epsilon * (repulsive + attractive)
        return energy


if __name__ == '__main__':
    print("Hello")

    mol_dyn = MolDynTwoAtom(r0_A=2.5,
                            epsilon=0.2,
                            r1_A=0,
                            r2_A=5,
                            v1_A_per_fs=0.1,
                            v2_A_per_fs=-0.1,
                            dt_fs=1.0,
                            grad_delta_A=0.01)
    for _ in range(10):
        mol_dyn.timestep()
