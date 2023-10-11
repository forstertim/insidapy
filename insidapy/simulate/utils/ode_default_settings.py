
import numpy as np

def load_default_settings(self, no_example):
    """
    Default initial conditions for the examples stored in the modules.

    Args:
        self (object, required): The 'self' object of the class.
        no_example (bool, required): A flag that will be set to 'False' if the example is identified.
    """

    # Copy the object

    # ..........................................
    if self.example == 'batch1':
        """ biomass consumes substrate and produces product
        """
        # System info
        no_example = False
        self.example_info = 'Batch fermentation with 3 species. Bacteria growth, substrate consumption and product formation. Mimics the production of a target protein.'
        self.example_reference = 'ISBN 0-13-512966-4'
        self.species = ['biomass', 'substrate', 'product']
        self.species_units = ['g/L', 'g/L', 'g/L']
        self.time_unit = 'h'
        # Bounds
        if self.overwrite_bounds_initial_conditions is None:
            self.LB = np.array([0.1, 50, 0])
            self.UB = np.array([0.4, 90, 0])
            text_to_print_bounds = 'default'
        # Time info
        if self.overwrite_bounds_time_span is None:
            self.tspan = [0,80]
            text_to_print_time = 'default'

    # ..........................................
    elif self.example == 'batch2':
        """ biomass consumes nitrate, carbon, and phosphate
        """
        # System info
        no_example = False
        self.example_info = 'Batch fermentation with 4 species. Bacteria growth, nitrate/carbon/phosphate consumption. Mimics a waste water treatment process.'
        self.example_reference = 'DOI 10.1002/bit.26881'
        self.species = ['biomass', 'nitrate', 'carbon', 'phosphate']
        self.species_units = ['mg/L', 'mg/L', 'mg/L', 'mg/L']
        self.time_unit = 'h'
        # Bounds
        if self.overwrite_bounds_initial_conditions is None:
            self.LB = np.array([216, 108, 450, 17])
            self.UB = np.array([264, 132, 550, 21])
            text_to_print_bounds = 'default'
        # Time info
        if self.overwrite_bounds_time_span is None:
            self.tspan = [0,180]
            text_to_print_time = 'default'

    # ..........................................
    elif self.example == 'batch3':
        """ E + S <-[k1],[ki1]-> ES ->[k2] E + P
        """
        # System info
        no_example = False
        self.example_info = 'Enzyme substrate interaction described by the Michaelis-Menten model. 4 species. E + S <-[k1],[ki1]-> ES ->[k2] E + P'
        self.example_reference = 'DOI 10.1021/acs.jpcb.2c08932'
        self.species = ['enzyme', 'substrate', 'complex', 'product']
        self.species_units = ['mmol/L', 'mmol/L', 'mmol/L', 'mmol/L']
        self.time_unit = 'min'
        # Bounds
        if self.overwrite_bounds_initial_conditions is None:
            self.LB = np.array([0.1, 1, 0, 0])
            self.UB = np.array([1, 10, 0, 0])
            text_to_print_bounds = 'default'
        # Time info
        if self.overwrite_bounds_time_span is None:
            self.tspan = [0,70]
            text_to_print_time = 'default'

    # ..........................................
    elif self.example == 'batch4':
        """ A -[k1]-> B -[k2]-> C
        """
        # System info
        no_example = False
        self.example_info = 'Series of reactions. 3 Species. A -[k1]-> B -[k2]-> C.'
        self.example_reference = 'DOI 10.1007/978-1-4757-3040-1'
        self.species = ['A', 'B', 'C']
        self.species_units = ['-', '-', '-'] # Example unitless in book
        self.time_unit = 's'
        # Bounds
        if self.overwrite_bounds_initial_conditions is None:
            self.LB = np.array([1e-3, 1e-3, 0])
            self.UB = np.array([1, 1, 0.2])
            text_to_print_bounds = 'default'
        # Time info
        if self.overwrite_bounds_time_span is None:
            self.tspan = [0,80]
            text_to_print_time = 'default'

    # ..........................................
    elif self.example == 'batch5':
        """ Nonisothermal Van de Vusse Reaction Case I
        A -[k1]-> B -[k2]-> C
        2 A -[k3]-> D
        """
        # System info
        no_example = False
        self.example_info = 'Van de Vusse reaction. 4 Species. \nA -[k1]-> B -[k2]-> C and 2 A -[k3]-> D.'
        self.example_reference = 'DOI 10.1007/978-1-4757-3040-1'
        self.species = ['A', 'B', 'C', 'D']
        self.species_units = ['mol/L', 'mol/L', 'mol/L', 'mol/L'] # Example unitless in book
        self.time_unit = 's'
        # Bounds
        if self.overwrite_bounds_initial_conditions is None:
            self.LB = np.array([1e-3, 1e-3, 0, 0])
            self.UB = np.array([1, 1, 0.2, 0.2])
            text_to_print_bounds = 'default'
        # Time info
        if self.overwrite_bounds_time_span is None:
            self.tspan = [0,4]
            text_to_print_time = 'default'

    # ..........................................
    elif self.example == 'fedbatch1':
        """ Bioreaction in fedbatch operation mode. 3 species and 1 volume. Mimics the production of a product by biomass while consuming substrate.
        """
        # System info
        no_example = False
        self.example_info = 'Bioreaction in fedbatch operation mode. Constant input flow. 3 species and 1 volume. Bacteria growth, substrate consumption and product formation. Mimics the production of a target protein.'
        self.example_reference = 'ISBN 978-1-119-28591-5'
        self.species = ['biomass', 'product', 'substrate', 'volume']
        self.species_units = ['g/L', 'g/L', 'g/L', 'L']
        self.time_unit = 'h'
        # Bounds
        if self.overwrite_bounds_initial_conditions is None:
            self.LB = np.array([0.01, 0, 5, 0.8])
            self.UB = np.array([0.1, 0, 15, 1.5])
            text_to_print_bounds = 'default'
        # Time info
        if self.overwrite_bounds_time_span is None:
            self.tspan = [0,50]
            text_to_print_time = 'default'
            
    return self, no_example