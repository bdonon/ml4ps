.. _acceleration:

Accelerating the Training Loop
==============================

For now, there are two main ways of accelerating the training loop : multiprocessing and converting the dataset
to pickle.

Multiprocessing
---------------

The torch implementation of data loaders allows to use multiprocessing, and to prefetch the next batch
before the end of the current training step. This can be done by setting the option `num_workers` of
the DataLoader to a non-zero integer.

.. code-block:: pycon

    loader = DataLoader(dataset,
                        batch_size=64,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=mp.power_grid_collate)

Pickle
------

It is also possible to convert a dataset into a series of .pkl files that can be "more easily" imported.
TODO : rendre le truc plus exhaustif

First, one needs to convert the dataset into a pickle version as follows :

.. code-block:: pycon

    train_dir = 'data/case118/train'
    train_di_pkl = 'data/case118/train_pkl'
    mp.pickle_dataset(train_dir, train_dir_pkl, backend=backend)

This creates a folder located at `data/case118/train_pkl` that contains pickle versions of all the data samples
contained in `data/case118/train`.
Then, these pickle versions can be imported on the fly by the PowerGridDataset and DataLoader.

.. code-block:: pycon

    dataset = mp.PowerGridDataset(data_dir=train_dir_pkl,
                                  backend=backend,
                                  normalizer=normalizer,
                                  pickle=True)
    loader = DataLoader(dataset,
                        batch_size=8,
                        shuffle=True,
                        collate_fn=mp.power_grid_collate)

.. module:: ml4ps.pickle
.. autofunction:: pickle_dataset

Loading in Memory
-----------------

If the dataset fits in RAM, then it is possible to fully load it before the start of the training loop.
This can greatly accelerate the training process.
One just needs to set the argument `load_in_memory` to `True` of the PowerGridDataset.
Also, early experiments show that disabling multiprocessing for the DataLoader provides a substantial speedup.

The `load_in_memory` option can be combined with the `pickle` option.

.. code-block:: pycon

    dataset = mp.PowerGridDataset(data_dir=train_dir_pkl,
                                  backend=backend,
                                  normalizer=normalizer,
                                  load_in_memory=True)
    loader = DataLoader(dataset,
                        batch_size=8,
                        shuffle=True,
                        collate_fn=mp.power_grid_collate)
