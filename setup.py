#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='cloudmlmagic',
      version='0.1.0dev',
      description='Jupyter Notebook Magics for Google Cloud ML Engine',
      author='Hayato Yoshikawa',
      url='https://github.com/hayatoy/cloudml-magic',
      packages=find_packages(),
      install_requires=['google-api-python-client', 'google-cloud-logging'],
      license="MIT",
      )
