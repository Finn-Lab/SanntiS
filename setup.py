# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import atexit
import os
import sys
import json
from setuptools import setup
from setuptools.command.install import install

with open('emeraldbgc/pkg_info.json') as h:
    _info = json.load(h)

with open('README.txt') as h:
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
    python_requires = ">=3.7",
    include_package_data = True,
    package_data={'emeraldbgc': [
                            'models/hmm_lib/*',
                            'models/*',
                            'modules/*',
                            '*json',
                            'exclude.txt',
                            ]
    },
    entry_points = {
            'console_scripts': [
                                'emeraldbgc = emeraldbgc._cli:main',
                                'emerald_build_gb = emeraldbgc.build_gb:main',
                                ]
    },
    packages = find_packages(exclude=('tests', 'docs')),
)
