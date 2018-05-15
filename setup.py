import os
from setuptools import setup
from projectpredict._version import __version__


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


version = __version__

setup(
    name="projectpredict",
    version=version,
    author="Justin Tervala",
    author_email="jgtervala@gmail.com",
    description="A library to help schedule projects intelligently.",
    long_description_content_type='text/markdown',
    license="MIT",
    keywords=["scheduling", "project management"],
    url="https://github.com/JustinTervala/ProjectPredict",
    download_url='https://github.com/JustinTervala/ProjectPredict/archive/v{}.tar.gz'.format(version),
    packages=['projectpredict'],
    long_description=read('README.md'),
    install_requires=[
        'scipy',
        'pandas',
        'matplotlib>=2.0.0',
        'networkx>=2.0',
        'scikit-learn>=0.18',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Topic :: Office/Business :: Scheduling',
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
)