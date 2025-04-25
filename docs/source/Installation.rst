Installation
==============

This document provides instructions for installing the thompson package.

Requirements
==============

The thompson package requires the following dependencies:

* Python 3.7 or higher
* numpy
* pandas
* matplotlib
* requests

Pypi Installation
==================

The easiest way to install thompson is via PyPI:

.. code-block:: bash

    pip install thompson

To force an update to the latest version:

.. code-block:: bash

    pip install -U thompson

Github Installation
=====================

To install the latest development version directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/erdogant/thompson.git

Development Installation
=========================

To install thompson in development mode:

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/erdogant/thompson.git
    cd thompson

2. Install in editable mode:

.. code-block:: bash

    pip install -e .

Uninstalling
=============

To remove the thompson package:

.. code-block:: console

    pip uninstall thompson

Note that this will only remove the thompson package itself. If you want to remove all dependencies as well, you'll need to uninstall them separately.

Verifying Installation
=======================

To verify that thompson is installed correctly, you can run:

.. code-block:: python

    import thompson as th
    print(th.__version__)

This should print the version number of your thompson installation.

.. include:: add_bottom.add