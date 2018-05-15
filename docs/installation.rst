.. _installation:

Installation
============
The easiest way to install ProjectPredict is to install it from PyPI using pip

.. code-block:: bash

    pip install projectpredict

Or, using `Pipenv <https://docs.pipenv.org>`_, the new officially recommended standard for Python package management,

.. code-block:: bash
   pipenv install projectpredict

Development Installation
------------------------

Currently the only way to install ProjectPredict for development is to clone it from GitHub.

.. Link to GitHub

.. code-block:: bash

    git clone https://github.com/JustinTervala/ProjectPredict


Set up your virtual environment using `virtualenv <https://virtualenv.pypa.io/en/stable/>`_

.. code-block:: bash

    git clone https://github.com/JustinTervala/ProjectPredict
    cd ProjectPredict
    virtualenv venv
    source venv/bin/activate

Then install the requirements

.. code-block:: bash

    pip install -r requirements.txt
    pip install -r requirements-dev.txt


Or, using Pipenv

.. code-block:: bash

    git clone https://github.com/JustinTervala/ProjectPredict
    cd ProjectPredict
    pipenv install --dev
    pipenv shell


Testing
-------

ProjectPredict uses pytest as its unit testing framework. You can run the tests from the top-level directory by simply
typing "pytest"

.. code-block:: bash

    pytest --cov=projectpredict

Building the Documentation
--------------------------

ProjectPredict uses `sphinx <http://www.sphinx-doc.org/en/master/>`_ to build the docs, and uses several plugins. From
the top-level directory,

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

This will generate the file in docs/_build/index.html. This file is the entry point to the documentation