.. _powergriddataset:

Power Grid Dataset
==================

We choose to build our implementation upon the `pytorch <https://pytorch.org>`_ data tools,
which provides many great features.
We refer users to `data API <https://pytorch.org/docs/stable/data.html#
single-and-multi-process-data-loading>`_ of pytorch for more information.

Power Grid Database
-------------------

As we want this tool to be easily accessible to the Power Systems community, it is compatible with standard
power grid formats. This is notably performed by relying on third-party libraries for
the loading of power grids, their modifications, and their simulations. We refer readers to the section
:ref:`Backend <backend>` for more information about this. Indeed, the file formats that can be read do
indeed depend on the backend choice (e.g. using `pandapower <http://www.pandapower.org>`_ as backend will
only allow you to read data that are readable by this package).

Moreover, this makes the comparison of Machine Learning techniques with other approaches much easier, as there is
no longer a need for data conversion from usual power grid formats to standard machine learning ones.

.. note::

    For now, there is no support for power grid time series. Please contact us if you would be interested
    in such a feature.

Dataset Instance
----------------

A dataset is a folder that contains all your power grid data sample files in the right format.
See `this dataset example <https://doi.org/10.5281/zenodo.7077699>`_ for instance.

Usage
-----

Backend definition
__________________

In order to create a power grid dataset from which we will iteratively draw samples, you first need to
define a backend. As explained in :ref:`Backend <backend>`, it is in charge of reading power grid files,
extracting features, modifying them and running power flow simulations.

.. code-block:: pycon

    import ml4ps as mp
    backend = mp.PandaPowerBackend()

Instantiating a dataset
_______________________

You may now instantiate a power grid dataset (which is a subclass of `torch.utils.data.Dataset`). Note that
you need to define two distinct datasets for the train and test sets.

.. code-block:: pycon

    train_dir = 'data/case60/train'
    train_set = mp.PowerGridDataset(data_dir=train_dir, backend=backend, normalizer=normalizer)

.. note::

    We recommend to apply a normalization to the power grid feature, as detailed in
    :ref:`Normalization <normalization>`. A normalizer can be passed to the
    PowerGridDateset constructor, so that data will be normalized at each call :

    .. code-block:: pycon

        normalizer = mp.Normalizer(data_dir=train_dir, backend=backend)
        train_set = mp.PowerGridDataset(data_dir=train_dir, backend=backend, normalizer=normalizer)

Splitting train and validation sets
___________________________________

It is possible to split a train set into a validation and a training ones, as follows :

.. code-block:: pycon

    train_set, val_set = torch.utils.data.random_split(train_set, [900, 100])

Note that the code above only works if the train_set is initially of size 1000. Lengths should be
adjusted manually.

Iterating over a dataset
________________________

Once defined, we can sample from our dataset using a torch DataLoader.

.. code-block:: pycon

    from torch.utils.data import DataLoader
    loader = DataLoader(train_set,
                        batch_size=64,
                        shuffle=True,
                        collate_fn=mp.collate_power_grid)

.. note::

    Please use our implementation of the collate function. As we iterate over power grid instances that
    are not tensors, this custom function only collates data contained in `a` and `x`, and returns
    the batch list of power grid instances.

Then, we can iterate from our data loader. At each iteration, we will obtain a tuple `(a, x, nets)`.
`a` contains the objects addresses, `x` contains the features and `nets` contains the power grid
instances built using the power package specified in the backend. The first two can be passed to a neural
network to perform some prediction, while the last one can be used to directly interact with power grids
(i.e. modify features, run power grid simulations and extract results).

.. code-block:: pycon

    for a, x, nets in loader:
        train_step(a, x, nets)

Contents
--------
.. module:: ml4ps.dataset
.. autoclass:: PowerGridDataset
