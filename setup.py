from setuptools import setup, find_packages
import pathlib

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(  name = 'insidapy',
        
        version = '0.2.9',
        
        description = 'Module for in-silico data generation in python.',
        long_description = long_description, 
        long_description_content_type = "text/markdown",
        
        author = 'Tim Forster',
        license = 'MIT',
        
        packages = find_packages(),
        
        project_urls = {
            "Homepage": "https://github.com/forstertim/insidapy",
            "Documentation": "https://insidapy.readthedocs.io/en/latest/?badge=latest",
            "Issues": "https://github.com/forstertim/insidapy/issues"
        },
                
        # package_data = {'': ['<subfolder>/*.dat']},
        # package_dir={PACKAGE_NAME: SOURCE_DIRECTORY},
        include_package_data = True,
        install_requires = [    'numpy',
                                'scipy',
                                'openpyxl',
                                'matplotlib',
                                'scikit-learn',
                                'prettytable',
                            ],
        
        zip_safe = False
    
    )
