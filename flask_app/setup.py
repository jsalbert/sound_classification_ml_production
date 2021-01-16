#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='sound-classification',
    version='1.0.0',
    description='Sound Classification App',
    author='jsalbert',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
)
