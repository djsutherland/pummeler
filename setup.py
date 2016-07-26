try:
    from setuptools import setup
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

import numpy as np
import os
import versioneer


setup(
    name='pummeler',
    author='Dougal J. Sutherland',
    author_email='dougal@gmail.com',
    url="https://github.com/dougalsutherland/pummeler/",
    packages=[
        'pummeler',
    ],
    description="Utilities for processing and analyzing ACS PUMS files.",
    install_requires=[
        'numpy',
        'pandas',
        'progressbar2',
    ],
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)

