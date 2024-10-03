.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://img.shields.io/conda/vn/conda-forge/MacroStat.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/MacroStat
    .. image:: https://pepy.tech/badge/MacroStat/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/MacroStat
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/MacroStat


.. image:: https://api.cirrus-ci.com/github/KarlNaumann/MacroStat.svg?branch=master
     :alt: Built Status
     :target: https://cirrus-ci.com/github/KarlNaumann/MacroStat
.. image:: https://img.shields.io/pypi/v/MacroStat.svg
  :alt: PyPI-Server
  :target: https://pypi.org/project/MacroStat/
.. image:: https://readthedocs.org/projects/macrostat/badge/?version=stable
    :target: https://macrostat.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status
.. image:: https://img.shields.io/coveralls/github/KarlNaumann/MacroStat/main.svg
     :alt: Coveralls
     :target: https://coveralls.io/r/KarlNaumann/MacroStat
.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

=========
MacroStat
=========


    A Package providing multiple tools for the statistical analysis and treatment of macroeconomic simulation models, with a particular focus on Agent-based and Stock-Flow Consistent Models


The purpose of this project is to provide a statistical toolbox for the analysis of Agent-based Models. The toolbox is developed in `python` and aims to provide a simple interface for researchers to attach their model, such that simulations can be steered from within the toolbox and the relevant analysis can be run (such as sensitivities, confidence intervals, and simulation studies). Only the analysis itself requires `python`, while the models can be written in any language.

The code was developed using Python 3.10. Backwards compatibility is not guaranteed

.. _pyscaffold-notes:

Roadmap
=======
Some features that are currently under development:

1. **diff**: Implementation of numerical differentiation using parallel processing and a wrapper for PyTorch to compute Jacobians and Hessians
2. **sample**: Implementation of further common samplers (e.g. MC, MCMC)
3. **analysis**: Implementation of basic statistical tests following Vandin et al. `"Automated and distributed statistical analysis of economic agent-based models" <https://www.sciencedirect.com/science/article/pii/S0165188922001634>`_
4. **models**: Starting a library of basic macroeconomic models such as the basic models from https://macrosimulation.org/, the SFC models of Godley & Lavoie, and agent-based models such as `Mark-0 <https://github.com/KarlNaumann/Mark0>`_
5. **explore**: Implementation of parameter-space analysis approaches such as the alogirthm of `Naumann-Woleske et al. (2024) <https://arxiv.org/abs/2111.08654>`_


Installation
============

This project requires Python v3.10 or later.

To install the latest version of the package from PyPI::

    pip install macrostat

Or, directly from GitHub::

   pip install git+https://github.com/KarlNaumann/MacroStat.git#egg=macrostat

If you'd like to contribute to the package, please read the CONTRIBUTING.md guide.

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
