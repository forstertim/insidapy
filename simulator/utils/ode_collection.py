"""
A collection of ODE systems that can be simulated by default

Author: Tim Forster, 20. Sep. 2023
"""

##########################################################################################
import numpy as np


##########################################################################################
def ODE_Batch_Fermentation_ThreeSpecs(y, t):
    '''ODE model for batch fermentation. Literature: Turton, Shaeiwtz, Bhattacharyya, Whiting \
        "Analysis, synthesis and design of chemical processes", Prentice Hall. 2018.
        ISBN 0-13-512966-4

    Args:
        y (array): Concentration of species of shape [n,].
        t (scalar): time.

    Returns:
        array: dydt - Derivative of the species of shape [n,].
    '''

    # Variables  
    X = y[0]
    S = y[1]
    P = y[2]

    # Parameters
    mu_max = 0.25; 	#h^-1
    K_S = 105.4;    #kg/m^3
    Y_XS = 0.07;    #[-]
    Y_PS = 0.167;   #[-]

    KXmu = 121.8669;#g/L constant for inhibition of biomass growth caused by higher cell densities

    T = 273 + 35;    #K
    R = 0.0083145;  #kJ/(K*mol) universial gas constant

    k1_ = 130.0307;  #[-] constant for activation of biomass growth
    E1_ = 12.4321; 	#kJ/mol activation enthalpy for biomass growth
    k2_ = 3.8343e48; #[-] constant for inactivation of biomass growth
    E2_ = 298.5476;	#kJ/mol inactivation enthalpy for biomass growth
    
    # Define temperature dependency of the rate constants
    k1 = k1_ * np.exp(-E1_ /(R*T))
    k2 = k2_ * np.exp(-E2_ /(R*T))

    # Calculate specific growth rate of the biomass
    mu = (mu_max*S)/(K_S+S) * k1/(1+k2) * (1-(X/(KXmu+X)))

    # Calculate consumption of substrate
    sigma = -(1/Y_XS)*mu

    # Calculate specific production rate of the protein
    pi = Y_PS/Y_XS*mu

    # Vectorization of rates
    rate = np.hstack((mu.reshape(-1,1), sigma.reshape(-1,1), pi.reshape(-1,1)))        

    # ODEs of biomass, volume and product  
    dydt = rate * X.reshape(-1,1)

    # Return
    return dydt.reshape(-1,)


##########################################################################################
def ODE_Batch_Fermentation_FourSpecs(y, t):
    
    '''ODE model for batch fermentation. Literature: Ehecatl Antonio Del Rio‐Chanona, Xiaoyan Cong, Eric Bradford, Dongda Zhang, Keju Jing.
    Review of advanced physical and data‐driven models fordynamic bioprocess simulation: Case study of algae–bacteriaconsortium wastewater treatment,
    Biotechnology & Bioengineering, 2018

    Args:
        y (array): Concentration of species of shape [n,].
        t (scalar): time.

    Returns:
        array: dydt - Derivative of the species of shape [n,].
    '''

    # Variables  
    X = y[0]/1e3
    N = y[1]
    C = y[2]
    P = y[3]

    # Parameters
    high_nutrient = True
    
    if high_nutrient:
        mu = 0.109          # 1/h
        K_N = 0.00860       # mg/L
        K_C = 0             # mg/L
        K_P = 0             # mg/L
        mu_d = 0.0854       # L/(g h)
        
        Y_C1 = 217          # mg/g
        Y_C2 = 0.839        # mg/(g h)
        
        Y_N1 = 5.36         # mg/g
        Y_N2 = 0.0559       # mg/(g h)
        
        Y_P1 = 2.74         # mg/g
        Y_P2 = 0.00833      # mg/(g h)
    else:
        mu = 0.0821         # 1/h
        K_N = 0.00873       # mg/L
        K_C = 0             # mg/L
        K_P = 0.001         # mg/L
        mu_d = 0.103        # L/(g h)
        
        Y_C1 = 85.5         # mg/g
        Y_C2 = 0.172        # mg/(g h)
        
        Y_N1 = 4.36         # mg/g
        Y_N2 = 0.0132       # mg/(g h)
        
        Y_P1 = 2.47         # mg/g
        Y_P2 = 0.00373      # mg/(g h)
    
    # Rate expressions
    dXdt = mu * (N / (N + K_N)) * (C / (C + K_C)) * (P / (P + K_P)) * X - mu_d * (X**2)
    dNdt = -Y_N1 * (mu * dXdt) - Y_N2 * X
    dCdt = -Y_C1 * (mu * dXdt) - Y_C2 * X
    dPdt = -Y_P1 * (mu * dXdt) - Y_P2 * X

    # Vectorization
    dydt = np.array((dXdt*1e3, dNdt, dCdt, dPdt))

    # Return
    return dydt.reshape(-1,)


##########################################################################################
def ODE_Batch_Michaelis_Menten_FourSpecs(y, t):
    
    '''ODE model for The Michaelis-Menten Kinetics. 
    The Michaelis-Menten model, which originated from the pioneering work of Michaelis and Menten in invertase experiments, 
    has been foundational for studies of enzyme catalysis. The corresponding mechanism describes for enzymatic reactions.
    The following system is implemented: E + S <-[k1],[ki1]-> ES -[k2]-> E + P

    Data from here: Samuel W. K. Wong, Shihao Yang, and S. C. Kou,
    Estimating and Assessing Differential Equation Models with Time-Course Data.
    J Phys Chem B. 2023.

    Args:
        y (array): Concentration of species of shape [n,].
        t (scalar): time.

    Returns:
        array: dydt - Derivative of the species of shape [n,].
    '''

    # Variables  
    E = y[0]    # mM
    S = y[1]    # mM
    ES = y[2]   # mM
    P = y[3]    # mM
    
    # Parameters
    k1 = 0.9    # (min mM)^-1
    ki1 = 0.75  # min^-1
    k2 = 2.54   # min^-1
    
    # Rate expressions
    dEdt = -k1*E*S + (ki1 + k2)*ES
    dSdt = -k1*E*S + ki1*ES
    dESdt = k1*E*S - (ki1 + k2)*ES
    dPdt = k2*ES

    # Vectorization
    dydt = np.array((dEdt, dSdt, dESdt, dPdt))

    # Return
    return dydt.reshape(-1,)


##########################################################################################
def ODE_Batch_Chemical_Reaction_1(y, t):
    
    '''ODE model for the following series of reactions: A -[k1]-> B -[k2]-> C

    Data from here: Floudas A. et al., 
    Handbook of Test Problems in Local and Global Optimization. In: Nonconvex Optimization and Its Applications.
    p. 113, Springer US, 1999.
    ISBN 978-1-4419-4812-0
    DOI 10.1007/978-1-4757-3040-1

    Args:
        y (array): Concentration of species of shape [n,].
        t (scalar): time.

    Returns:
        array: dydt - Derivative of the species of shape [n,].
    '''

    # Variables  
    A = y[0]    # Unitless
    B = y[1]    # Unitless
    C = y[2]    # Unitless
    
    # Parameters
    k1 = 0.09755988     # s^-1
    k2 = 0.09658428     # s^-1

    # Rate expressions
    dAdt = -k1*A
    dBdt = k1*A - k2*B
    dCdt = k2*B

    # Vectorization
    dydt = np.array((dAdt, dBdt, dCdt))

    # Return
    return dydt.reshape(-1,)


##########################################################################################
def ODE_Batch_Chemical_Reaction_2(y, t):    
    '''ODE model for the Van de Vusse Reaction Case I: 
    A -[k1]-> B -[k2]-> C
    2A -[k3]-> D

    Data from here: Floudas A. et al., 
    Handbook of Test Problems in Local and Global Optimization. In: Nonconvex Optimization and Its Applications.
    p. 139, Springer US, 1999.
    ISBN 978-1-4419-4812-0
    DOI 10.1007/978-1-4757-3040-1

    Args:
        y (array): Concentration of species of shape [n,].
        t (scalar): time.

    Returns:
        array: dydt - Derivative of the species of shape [n,].
    '''

    # Variables  
    A = y[0]    # mol/L
    B = y[1]    # mol/L
    C = y[2]    # mol/L
    D = y[3]    # mol/L
    
    # Parameters
    k1 = 10     # s^-1
    k2 = 1      # s^-1	
    k3 = 0.5    # L(mol s)

    # Rate expressions
    dAdt = -k1*A - 2*k3*(A**2)
    dBdt = k1*A - k2*B
    dCdt = k2*B
    dDdt = k3*(A**2)

    # Vectorization
    dydt = np.array((dAdt, dBdt, dCdt, dDdt))

    # Return
    return dydt.reshape(-1,)


##########################################################################################
def fedbatch1(y, t):
    '''ODE model for a biorector in fedbatch operation mode. 
    Bacteria growth, substrate consumption and product formation. 
    Mimics the production of a target protein.'

    Data from here: Seborg, Edgar, Mellichamp, Doyle, 
    Process Dynamics and Control, 4th edition.
    section 2.4.8, 2016, Wiley.
    ISBN: 978-1-119-28591-5
    
    Jupyter example: https://dynamics-and-control.readthedocs.io/en/latest/1_Dynamics/2_Time_domain_simulation/Fed%20batch%20bioreactor.html

    Args:
        y (array): Concentration of species of shape [n,].
        t (scalar): time.

    Returns:
        array: dydt - Derivative of the species of shape [n,].
    '''
    
    # Variables
    X = y[0]
    P = y[1]
    S = y[2]
    V = y[3]

    # Parameters
    mumax = 0.20      # 1/hour
    Ks = 1.00         # g/liter
    Yxs = 0.5         # g/g
    Ypx = 0.2         # g/g
    Sf = 10.0         # g/liter
    
    # inlet flowrate for each time
    def F(t):
        return 0.05
    
    # Calculated variables
    def mu(S):
        return mumax*S/(Ks + S)
    def Rg(X,S):
        return mu(S)*X
    def Rp(X,S):
        return Ypx*Rg(X,S)
    
    # Rate expressions
    dXdt = -F(t)*X/V + Rg(X,S)
    dPdt = -F(t)*P/V + Rp(X,S)
    dSdt = F(t)*(Sf-S)/V - Rg(X,S)/Yxs
    dVdt = F(t)

    # Vectorization
    dydt = np.array((dXdt, dPdt, dSdt, dVdt))

    # Return
    return dydt.reshape(-1,)