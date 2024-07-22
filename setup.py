"""
Setup script for mri_preprocess tool.
"""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('mri_preprocess', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.md') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name='mri_preprocess',
    version=__version__,
    description='Carries out basic preprocessing for MRI data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.9',
    author='Niall Duncan',
    author_email='niall.w.duncan@gmail.com',
    url='https://github.com/nw-duncan/mri-preprocess',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    platforms='any',
    keywords=['neuroscience', 'anatomical', 'MRI', 'functional', 'preprocessing'],
    install_requires=install_requires,
    include_package_data=True,
)