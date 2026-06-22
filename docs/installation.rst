Installation
------------

Prerequisites
~~~~~~~~~~~~~~

Install python 3.10 or 3.11 (**recommended**).
Astir has been tested on Python 3.9-3.11.


Astir installation
~~~~~~~~~~~~~~~~~~
PyPI
####

.. code::

    pip3 install astir

Conda
####

The source code provides a conda environment templates under `envs/astir.yml`:

.. code::

    git clone https://github.com/camlab-bioml/astir.git
    conda env create -f envs/astir.yml
    pip install -e .

Dev
####
With one of the recommended Python versions, clone this repo and run

.. code::

    git clone https://github.com/camlab-bioml/astir.git
    cd astir
    pip install -e .

