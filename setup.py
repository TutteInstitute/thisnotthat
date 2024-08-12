#! /usr/bin/env python
import codecs
import os
from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('thisnotthat', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'thisnotthat'
DESCRIPTION = 'Tools for interactive visualization and exploration of _data maps_'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Benoit Hamelin, John Healy, Leland McInnes'
MAINTAINER_EMAIL = 'leland.mcinnes@gmail.com'
URL = 'https://github.com/TutteInstitute/thisnotthat'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/TutteInstitute/thisnotthat'
VERSION = __version__
INSTALL_REQUIRES = [
    "bokeh",
    "cmocean",
    "colorcet",
    "glasbey",
    "hdbscan",
    "matplotlib",
    "numpy>=1.22",
    "pandas",
    "panel",
    "param",
    "scikit-learn>=1.2.0",
    "umap-learn",
    "vectorizers",
    "wordcloud",
]
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.9',
               'Programming Language :: Python :: 3.10'
               ]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov',
        'psutil',
        'pytest-asyncio',
        'pytest-rerunfailures',
        'pytest-xdist',
        'pytest-playwright'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'nbsphinx',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
