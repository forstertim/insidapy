from setuptools import setup, find_packages

setup(  name = 'simulator',
        
        version = '0.0.14',
        
        description = 'Module for generating example data.',
        
        author = 'Tim Forster',
        
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
