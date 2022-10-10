from distutils.core import setup

from setuptools import find_packages

import os

# Optional project description in README.md:

current_directory = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception: long_description = ''

setup(

# Project name:

name='NTNU MASCOT',

# Packages to include in the distribution:

packages=find_packages(','),

# Project version number:

version='0.0.2',

# List a license for the project, eg. MIT License

license='MIT',

# Short description of your library:

description='Test MASCOT',

# Long description of your library:

long_description=long_description,

long_description_content_type='text/markdown',

# Your name:

author='Yaolin Ge',

# Your email address:

author_email='geyaolin@gmail.com',

# Link to your github repository or website:

url='',

# Download Link from where the project can be downloaded from:

download_url='',

# List of keywords:

keywords=['autonomous ocean sampling'],

# List project dependencies:

install_requires=[
'numpy',
'matplotlib',
'plotly',
'shapely',
'scipy',
'pandas'
],

# python_requires="==3.9.12",
# https://pypi.org/classifiers/

classifiers=[]

)
