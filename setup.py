import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="projectpredict",
    version="0.0.1",
    author="Justin Tervala",
    author_email="jgtervala@gmail.com",
    description="A library to help schedule projectes intelligently.",
    license="MIT",
    keywords="project management",
    url="https://github.com/JustinTervala/ProjectPredict",
    packages=['projectpredict'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Topic :: Office/Business :: Scheduling',
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
)