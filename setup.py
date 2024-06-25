from setuptools import setup, find_packages

setup(  name = 'insidapy',
        
        version = '0.2.6',
        
        description = 'Module for in-silico data generation in python.',
        
        author = 'Tim Forster',
        
        license = 'MIT',
        
        packages = find_packages(),
        
        # package_data = {'': ['<subfolder>/*.dat']},
        # package_dir={PACKAGE_NAME: SOURCE_DIRECTORY},
        
        install_requires = [    'numpy',
                                'scipy',
                                'matplotlib',
                                'scikit-learn',
                                'prettytable',
                            ],
        
        zip_safe = False
    
    )
