import numpy as np

def customode(y, t, coefs):
        """Custom ODE system. A batch reactor is modeled with two species. The following system
        is implemented: A <-[k1],[k2]-> B -[k3]-> C

        Args:
            y (array): Concentration of species of shape [n,].
            t (scalar): time.
            coefs (dict): Dictionary of coefficients or other information.

        Returns:
            array: dydt - Derivative of the species of shape [n,].
        """
    
        # Variables  
        A = y[0]
        B = y[1]
        C = y[2]

        # Parameters
        k1 = coefs['k1']
        k2 = coefs['k2']
        k3 = coefs['k3']    

        # Rate expressions
        dAdt = k2*B - k1*A
        dBdt = k1*A - k2*B - k3*B
        dCdt = k3*B

        # Vectorization
        dydt = np.array((dAdt, dBdt, dCdt))

        # Return
        return dydt.reshape(-1,)