import os
from setuptools import setup
from projectpredict import __version__

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="projectpredict",
    version=__version__ + 'a1',
    author="Justin Tervala",
    author_email="jgtervala@gmail.com",
    description="A library to help schedule projectes intelligently.",
    license="MIT",
    keywords=["scheduling", "project management"],
    url="https://github.com/JustinTervala/ProjectPredict",
    #download_url='https://github.com/JustinTervala/ProjectPredict/archive/v{}.tar.gz'.format(__version__),
    packages=['projectpredict'],
    long_description=read('README.md'),
    install_requires=[
        'scipy',
        'networkx',
        'scikit-learn',
        'matplotlib',
        'pandas'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Topic :: Office/Business :: Scheduling',
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
)