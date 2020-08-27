===================================================================================
astir - Automated cell identity from single-cell multiplexed imaging and proteomics
===================================================================================

|Build Status| |PyPI| |Code Style|

.. |Build Status| image:: https://travis-ci.org/camlab-bioml/astir.svg?branch=master
    :target: https://travis-ci.org/camlab-bioml/astir
.. |Code Style| image:: https://img.shields.io/badge/code%20style-black-black
    :target: https://github.com/python/black
.. |PyPI| image:: https://img.shields.io/badge/pypi-v2.1-orange
    :target: https://pypi.org/project/pypi/


``astir`` is a modelling framework for the assignment of cell type and cell state across a range of single-cell technologies such as Imaging Mass Cytometry (IMC). ``astir`` is built using `pytorch <https://pytorch.org/>`_ and uses recognition networks for fast minibatch stochastic variational inference. 

.. image:: https://www.camlab.ca/img/astir.png
    :align: center
    :alt: automated single-cell pathology

Key Applications
---------------------

- To predict cell types given cell expression data
- To predict cell states given cell expression data
-

Getting started
---------------------

See the full `documentation <https://astir.readthedocs.io/en/latest>`_ and check out the `tutorials <https://astir.readthedocs.io/en/latest/tutorials/index.html>`_.


Authors
---------------------

| Jinyu Hou, Sunyun Lee, Michael Geuenich, Kieran Campbell
| Lunenfeld-Tanenbaum Research Institute & University of Toronto
