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

    loader = DataLoader(train_set,
                        batch_size=64,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=mp.power_grid_collate)

Pickle
------

It is also possible to convert a dataset into a series of .pkl files that can be "more easily" imported.


.. module:: ml4ps.pickle
.. autofunction:: pickle_dataset

