# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import atexit
import os
import sys
import json
from setuptools import setup
from setuptools.command.install import install

with open('sanntis/pkg_info.json') as h:
    _info = json.load(h)

with open('README.md') as h:
    readme = h.read()

setup(
    name = _info['name'],
    version = _info['version'],
    description = _info['description'],
    author = _info['author'],
    author_email = _info['author_email'],
    licence = _info.get('licence'),
    url = _info['url'],
    long_description = readme,
    python_requires = ">=3.9",
    include_package_data = True,
    package_data={'sanntis': [
                            'models/hmm_lib/*',
                            'models/*',
                            'modules/*',
                            '*json',
                            'exclude.txt',
                            ]
    },
    entry_points = {
            'console_scripts': [
                                'sanntis = sanntis._cli:main',
                                'sanntis_build_gb = sanntis.build_gb:main',
                                ]
    },
    packages = find_packages(exclude=('tests', 'docs')),
)
