.. _normalization:

Normalization
=============

Neural networks can only work properly if your data is properly normalized.
In most cases, a simple standardization (subtracting the mean and dividing
by the standard deviation) is enough.
However, data encountered in power grids is quite atypical and is very likely
to display multimodal distributions.

Moreover, we wish to have a normalization process that does not alter the
permutation-equivariance of the data. For more details about some properties
of our data, please refer to :ref:`Data Formalism <data-formalism>`.

Explanation
-----------

Here we explain that we want to use an approximation of the cdf.

Usage
-----

.. code-block:: console

    x_norm = normalizer(x)

Advanced Options
----------------

If you want to do some fancy stuff, you can do this and this.

Contents
--------
.. module:: ML4PS.normalization
.. autoclass:: Normalizer