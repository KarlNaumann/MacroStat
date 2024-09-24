.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/MacroStat.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/MacroStat
    .. image:: https://readthedocs.org/projects/MacroStat/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://MacroStat.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/MacroStat/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/MacroStat
    .. image:: https://img.shields.io/pypi/v/MacroStat.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/MacroStat/
    .. image:: https://img.shields.io/conda/vn/conda-forge/MacroStat.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/MacroStat
    .. image:: https://pepy.tech/badge/MacroStat/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/MacroStat
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/MacroStat

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=========
MacroStat
=========


    A Package providing multiple tools for the statistical analysis and treatment of macroeconomic simulation models, with a particular focus on Agent-based and Stock-Flow Consistent Models


The purpose of this project is to provide a statistical toolbox for the analysis of Agent-based Models. The toolbox is developed in `python` and aims to provide a simple interface for researchers to attach their model, such that simulations can be steered from within the toolbox and the relevant analysis can be run (such as sensitivities, confidence intervals, and simulation studies). Only the analysis itself requires `python`, while the models can be written in any language. 

The code was developed using Python 3.10. Backwards compatibility is not guaranteed

.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd MacroStat
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Don't forget to tell your contributors to also install and use pre-commit.

.. _pre-commit: https://pre-commit.com/

Contact
=======

Karl Naumann-Woleske - karlnaumann.com

Project Link: [https://github.com/KarlNaumann/MacroStat](https://github.com/KarlNaumann/MacroStat)

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
