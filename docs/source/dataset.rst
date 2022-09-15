.. _dataset:

Power Grid Dataset
==================

We choose to build our implementation upon the `pytorch <https://pytorch.org>`_ data tools,
which provides many great features.
We refer users to `data API <https://pytorch.org/docs/stable/data.html#
single-and-multi-process-data-loading>`_ of pytorch for more information.

Usage
-----



.. note::

    We recommend to apply a normalization to the power grid feature, as detailed in
    :ref:`Normalization <normalization>`. A normalizer can be passed to the
    PowerGridDateset constructor, so that data will be normalized at each call.

.. code-block:: pycon

    import ml4ps as mp
    backend = mp.PandaPowerBackend()
    train_dir = 'data/case14/train'
    normalizer = mp.Normalizer(data_dir=train_dir, backend=backend)
    train_set = mp.PowerGridDataset(data_dir=train_dir, backend=backend, normalizer=normalizer)

Once defined, we can sample from our dataset using a torch DataLoader.

.. code-block:: pycon

    from torch.utils.data import DataLoader
    loader = DataLoader(train_set,
                        batch_size=64,
                        shuffle=True,
                        collate_fn=mp.collate_power_grid)

.. note::

    Use our implementation of the collate function. As we iterate over power grid instances that
    are not tensors, this custom function only collates data contained in `a` and `x`, and returns
    the batch list of power grid instances.

Then, we can iterate from our data loader as follows :

.. code-block:: pycon

    for a, x, nets in loader:
        train_step(a, x, nets)

Accelerating the training loop
______________________________

The torch implementation of data loaders allows to use multiprocessing, and to prefetch the next batch
before the end of the current training step. This can be done by setting the option `num_workers` of
the DataLoader to a non-zero integer.

.. code-block:: pycon

    loader = DataLoader(train_set,
                        batch_size=64,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=mp.power_grid_collate)

Splitting train and validation sets
___________________________________

It is possible to split a train set into a validation and a training ones, as follows :

.. code-block:: pycon

    train_set, val_set = torch.utils.data.random_split(train_set, [900, 100])

Note that the code above only works if the train_set is initially of size 1000. Lengths should be
adjusted manually.

Contents
--------
.. module:: ML4PS.dataset
.. autoclass:: PowerGridDataset
