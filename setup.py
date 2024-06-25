from setuptools import setup, find_packages
import pathlib

setup(  name = 'insidapy',
        
        version = '0.2.6',
        
        description = 'Module for in-silico data generation in python.',
        long_description = pathlib.Path("README.md").read_text(),
        long_description_content_type = "text/markdown",
        
        author = 'Tim Forster',
        license = 'MIT',
        
        packages = find_packages(),
        
        project_urls = {
            "Documentation": "https://insidapy.readthedocs.io/en/latest/?badge=latest"
        },
                
        # package_data = {'': ['<subfolder>/*.dat']},
        # package_dir={PACKAGE_NAME: SOURCE_DIRECTORY},
        
        install_requires = [    'numpy',
                                'scipy',
                                'openpyxl',
                                'matplotlib',
                                'scikit-learn',
                                'prettytable',
                            ],
        
        zip_safe = False
    
    )
