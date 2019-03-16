from setuptools import setup, find_packages
import os

NAME = 'csai_test'
VERSION = '1.0'
REQUIRED_PACKAGES = ['keras>=2.2.4']


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    )