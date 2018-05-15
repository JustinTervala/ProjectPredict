# ProjectPredict
[![Build Status](https://travis-ci.org/JustinTervala/ProjectPredict.svg?branch=master)](https://travis-ci.org/JustinTervala/ProjectPredict)
[![Maintainability](https://api.codeclimate.com/v1/badges/888401889e781419c249/maintainability)](https://codeclimate.com/github/JustinTervala/ProjectPredict/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/888401889e781419c249/test_coverage)](https://codeclimate.com/github/JustinTervala/ProjectPredict/test_coverage)

ProjectPredict is a library to help project managers gain insight into the status of their project using Bayesian
networks. It is inspired by the paper ["Project scheduling: Improved approach to incorporate uncertainty using Bayesian
networks](https://www.pmi.org/learning/library/project-scheduling-approach-incorporate-uncertainty-2371)
(Khodakarami, Fenton, & Neil, Project Management Journal, 2007). The project features

* Inferring the latest start date, earliest finish date, and total float for each task in a project
* Recommending which task or tasks should be started next using custom constraints and objective functions
* Task duration specified either through 
  [three-point (PERT) estimation](https://en.wikipedia.org/wiki/Three-point_estimation) or inferring the duration of a 
  task from a custom machine learning model
* Visualization of a project timeline using [Matplotlib](https://matplotlib.org)

## Installation
The easiest way to install ProjectPredict is to install it from PyPI using pip

`pip install projectpredict`

## Documentation

More information can be found [here](http://projectpredict.readthedocs.io/en/latest/)