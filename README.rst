===================================================================================
astir - Automated cell identity from single-cell multiplexed imaging and proteomics
===================================================================================

|Build Status| |PyPI| |Code Style|

.. |Build Status| image:: https://travis-ci.com/camlab-bioml/astir.svg?branch=master
    :target: https://travis-ci.org/camlab-bioml/astir
.. |Code Style| image:: https://img.shields.io/badge/code%20style-black-black
    :target: https://github.com/python/black
.. |PyPI| image:: https://img.shields.io/badge/pypi-v2.1-orange
    :target: https://pypi.org/project/pypi/


``astir`` is a modelling framework for the assignment of cell type across a range of single-cell technologies such as Imaging Mass Cytometry (IMC). ``astir`` is built using `pytorch <https://pytorch.org/>`_ and uses recognition networks for fast minibatch stochastic variational inference. 

Key applications:

- Automated assignment of cell type and state from highly multiplexed imaging and proteomic data
- Diagnostic measures to check quality of resulting type and state inferences
- Ability to map new data to cell types and states trained on existing data using recognition neural networks
- A range of plotting and data loading utilities


.. image:: https://www.camlab.ca/img/astir.png
    :align: center
    :alt: automated single-cell pathology

Getting started
---------------------

Launch the interactive tutorial: 
<a href="https://colab.research.google.com/github/camlab-bioml/Astir-Vignette/blob/main/astir_tutorial.ipynb" target="_parent"><img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a> 
.. |On github| image:: https://img.shields.io/badge/on-github-black
    :target: https://github.com/camlab-bioml/Astir-Vignette


See the full `documentation <https://astir.readthedocs.io/en/latest>`_ and check out the `tutorials <https://astir.readthedocs.io/en/latest/tutorials/index.html>`_.


Authors
---------------------

| Jinyu Hou, Sunyun Lee, Michael Geuenich, Kieran Campbell
| Lunenfeld-Tanenbaum Research Institute & University of Toronto
