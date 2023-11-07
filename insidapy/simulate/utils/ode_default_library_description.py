"""
Default example information for ODE module.

"""
import numpy as np
from prettytable import PrettyTable

def ode_default_library_description(type_of_example):
    """Collection of all examples that are implemented in the ODE module.

    Args:
        type_of_example (str, required): Type of example (batch, fedbatch, custom, ...).
    """

    # Initialize empty dictionary
    examples = {}

    # Add every default example 
    # ..........................................
    if type_of_example == 'batch':
        examples['batch1'] = 'Batch fermentation with 3 species. Bacteria growth, substrate consumption and product formation. Mimics the production of a target protein.'
        examples['batch2'] = 'Batch fermentation with 4 species. Bacteria growth, nitrate/carbon/phosphate consumption. Mimics a waste water treatment process.'
        examples['batch3'] = 'Enzyme substrate interaction described by the Michaelis-Menten model. 4 species. E + S <-[k1],[ki1]-> ES ->[k2] E + P'
        examples['batch4'] = 'Series of reactions. 3 species. A -[k1]-> B -[k2]-> C.'
        examples['batch5'] = 'Van de Vusse reaction. 4 species. A -[k1]-> B -[k2]-> C and 2 A -[k3]-> D.'
        examples['batch6'] = 'Batch fermentation with 7 species: Biomass (total, viable, death cells), glucose, glutamine, lactate, and ammonia. Mimics CHO cell growth in reactor.'
    
    elif type_of_example == 'fedbatch':
        examples['fedbatch1'] = 'Bioreaction in fedbatch operation mode. Constant input flow. 3 species and 1 volume. Bacteria growth, substrate consumption and product formation. Mimics the production of a target protein.'
        
    # Create table with examples
    print(f'[+] The following examples are implemented in this {type_of_example.upper()} class:')
    x = PrettyTable()
    x.field_names = ["Example ID string", "Description"]
    x.align['Property'] = "l"
    x.align['Description'] = "l"
    x.min_width = 20
    x.max_width = 80
    
    # Add information rows
    for example_id_int, example_id_str in enumerate(examples):
        x.add_row([f'{example_id_str}', f'{examples[example_id_str]}'])
        
    # Display table
    print(x)
    print('\n')