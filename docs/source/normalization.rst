.. _normalization:

Normalization
=============

Neural networks can only work properly if your data is well distributed.
In most cases, a simple standardization (subtracting the mean and dividing
by the standard deviation) is more than enough.
However, data encountered in power grids is quite atypical and is very likely
to display multimodal distributions.

Moreover, we wish to have a normalization process that does not alter the
permutation-equivariance of the data. For more details about some properties
of our data, please refer to :ref:`Data Formalism <data-formalism>`.

Our data being composed of multiple instances of various classes (buses,
generators, loads, lines, etc.), we wish to have a normalizing mapping for
each class.
Even for a given class, there are multiple features that may be defined in
different units. For instance the active power of generators is usually
defined in `MW`, while the voltage setpoint of generators are usually defined
in `p.u.`.
Those quantities being defined in different units, it would make no sense
to use the same normalizing mapping for all features.

To sum things up, we need to build a normalizing function for each feature
of each class. As a consequence all active power of all generators will be
normalized using the exact same mapping.

.. note::

    One may argue that we could also use a different normalizing function
    for the different instances of a given class, the rationale being that,
    for instance, two different generators may produce very different power
    orders of magnitudes. Thus, using a separate normalizing function for
    each instance may also work.

    By doing so, we would actually break the permutation-equivariance of
    the data. If the neural network used is a simple fully connected
    architecture, then this may not have that much of an impact. But
    if we were to use a permutation-equivariant neural network architecture
    (such as a Graph Neural Network), then this would introduce a detrimental
    noise, which could prevent the neural network from learning anything
    meaningful.

Fitting normalizing functions
-----------------------------

Let us consider a single feature of a single class of objects (e.g. active
power of loads, expressed in `MW`). If we take a look at the distribution
of values across all objects of this class and across all power grid
instances (e.g. all active power of all loads of all power grids in a
given dataset), we may observe some atypical and multimodal distribution,
as illustrated in the figure below.
In this case, standardization is not enough to make data suitable for our
neural network. We are looking for another way of mapping this odd
distribution to a more appropriate one.

.. image:: figures/distribution.png
  :width: 400

Cumulative Distribution Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fortunately, the CDF (Cumulative Derivative Function) provides by definition
an efficient way of converting our data to a uniform law over the interval
[0, 1]. Moreover, for computational reasons, we may even want to
consider a subset of the empirical distribution (see `amount_of_samples`
parameter).

.. image:: figures/cdf.png
  :width: 400

Approximating the CDF
^^^^^^^^^^^^^^^^^^^^^

The empirical CDF has one major drawback, as it is made of discrete increments.
To solve this issue, we propose to build a piecewise linear approximation of
this function. To do so, we introduce a parameter `break_points` which define
the amount of breakpoints we want to have in our normalizing function.
We split the interval [0,1] into `break_points` equal chunks, and look at
the corresponding quantiles (displayed as red dots in the figure below).
We then use the linear interpolation provided by `scipy <https://scipy.org>`_.

.. image:: figures/approximation.png
  :width: 400

Merging equal quantiles
^^^^^^^^^^^^^^^^^^^^^^^

In general, it is possible that you obtain multiple equal quantiles. As a
result, the obtained interpolation is not continuous. In such a case, we
simply merge equal quantiles by taking the mean of the corresponding
probabilities. For instance, in the figure below, we merged the `20%`
and `40%` quantiles into a `30%` quantile.
The interpolation is now continuous.

.. image:: figures/conflicts.png
  :width: 400

Out of distribution extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since we only have access to a partial empirical distribution, it is very
likely that some values in the train and/or test sets will be out of the
range of observed values. If we only took the interpolation as it is,
then those values would all be mapped to either 0 or 1 (depending if it
is above or below the range of observed values). This would prevent
the neural network to make a distinction between values that are out of range.

Thus, we propose to extrapolate by extending the first and last slopes.
The rationale behind this choice is the following. Larger (resp. smaller)
values should have a very similar order of magnitude as the max (resp. min)
value that was used to fit the normalizing function. Since we want a
continuous and non-constant function, extending the largest (resp. smallest)
non-zero slope will map new values very close, disregard the data order of
magnitude. These extensions are illustrated in the figure below.

.. image:: figures/extrapolation.png
  :width: 400

Usage
-----

A normalizer can be built using a dataset :

.. code-block:: console

    import ml4ps as mp
    normalizer = mp.Normalizer(data_dir = data_dir, backend_name = 'pandapower')

Once built, it can normalize feature data provided by an interface.
See :ref:`Interface <interface>` for more information on how to get power system
data, and :ref:`Data Formalism <data-formalism>` for an explanation of the data
formalism.

.. code-block:: console

    x_norm = normalizer(x)

A normalizer can be saved into a `.pkl` file.

.. code-block:: console

    normalizer.save('my_normalizer.pkl')

It can then be loaded from the said `.pkl` file.

.. code-block:: console

    normalizer = mp.Normalizer('my_normalizer.pkl')

Contents
--------
.. module:: ML4PS.normalization
.. autoclass:: Normalizer
    :members: load, save
    :special-members: __init__, __call__
