.. title:: MacroStat Documentation
.. module:: MacroStat

========================================
MacroStat |release| Documentation
========================================

MacroStat is a package providing multiple tools for the creation, statistical analysis and treatment of macroeconomic simulation models, with a particular focus on Agent-based and Stock-Flow Consistent Models.

Installation
============

This project requires Python v3.10 or later.

.. tab-set::
    :class: sd-width-content-min

    .. tab-item:: pip

        .. code-block:: bash

            pip install macrostat

    .. tab-item:: GitHub

        .. code-block:: bash

            pip install git+https://github.com/KarlNaumann/MacroStat.git#egg=macrostat

Using MacroStat
===============

.. grid:: 1 1 2 2
    :gutter: 1

    .. grid-item::

        .. grid:: 1 1 1 1
            :gutter: 1

            .. grid-item-card::
               :padding: 2

               **Getting Started with MacroStat**
               ^^^

               The user guide provides in-depth information on MacroStat's key concepts, including model
               specification, simulation, and analysis. It includes detailed examples
               and best practices.

               .. toctree::
                  :maxdepth: 1

                  user_guide/index.rst

            .. grid-item-card::
               :padding: 2

               **API reference**
               ^^^

               The API reference contains detailed documentation of MacroStat's classes,
               methods, and functions.

               .. toctree::
                     :maxdepth: 1

                     api_reference.rst

    .. grid-item::

        .. grid:: 1 1 1 1
            :gutter: 1

            .. grid-item-card::
               :padding: 2

               **MacroStat's Model Library**
               ^^^

               Check out the model library for a collection of pre-built models. These include
               both Agent-based (ABM) and Stock-Flow Consistent (SFC) models.

               .. toctree::
                  :maxdepth: 2

                  models/index

What's new
==========

.. grid:: 1 1 2 2
    :class-row: sd-align-minor-center

    .. grid-item::
        :columns: 9

        Learn about new features and API changes.

    .. grid-item::
        :columns: 3

        .. toctree::
            :maxdepth: 1

            changelog.rst


Contribute
==========

.. grid:: 1 1 2 2
   :class-row: sd-align-minor-center

   .. grid-item::
      :columns: 9

      MacroStat is aimed to be a community project maintained for and by its users. See
      :doc:`develop/contributing` for the different ways that you can contribute to MacroStat!

   .. grid-item::
         :columns: 3

         .. rst-class:: section-toc
         .. toctree::
            :maxdepth: 2

            develop/index.rst

About MacroStat
===============

.. grid:: 1 1 2 2
    :class-row: sd-align-minor-center

    .. grid-item::
        :columns: 9

        MacroStat was created by Karl Naumann-Woleske to work with macroeconomic simulation models.
        Its goal was to make it easy for model developers to apply uniform statistical methods to their models,
        leverage the power of autodiff for the calibration of models, and to provide a flexible framework for the development of new models.

    .. grid-item::
        :columns: 3

        .. rst-class:: section-toc
        .. toctree::
            :maxdepth: 2

            project/index.rst


.. toctree::
   :maxdepth: 1
   :hidden:

   References <references>



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. _toctree: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: https://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain
.. _Sphinx: https://www.sphinx-doc.org/
.. _Python: https://docs.python.org/
.. _Numpy: https://numpy.org/doc/stable
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: https://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: https://scikit-learn.org/stable
.. _autodoc: https://www.sphinx-doc.org/en/master/ext/autodoc.html
.. _Google style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: https://www.sphinx-doc.org/en/master/domains.html#info-field-lists
