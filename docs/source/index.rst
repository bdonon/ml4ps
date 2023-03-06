ml4ps
=====

**ml4ps** (Machine Learning for Power Systems) is a Python library that facilitates the application of
Machine Learning to Power Systems, with a strong emphasis on respecting the data structure.

At the core of this package is the idea that
**real-life power systems have a structure that varies** through time (because of line disconnections,
unforeseen incidents, the building of new facilities). For this reason, we believe that developing
**AI models that respect the graph structure of power grid data** is critical to their application to
real-life systems.

We provide a series of tools that were built with this concern in mind :

    - a **data formalism** that properly describes the actual structure of the data ;
    - a **dataset** class derived from the `PyTorch <https://pytorch.org/docs/stable/data.html#
      single-and-multi-process-data-loading>`_ data loading utility, which returns objects
      that exhaustively describe power grid instances ;
    - a **normalizer** class that maps ill-distributed features into a more appropriate range of values ;
    - a **graph neural network** implementation in `JAX <https://jax.readthedocs.io/en/latest/>`_ that
      respects the structure of our power grid data ;
    - a **post-processing** class that sends the neural network output into physically meaningful orders
      of magnitudes ;
    - an **interface** that allows to plug various power grid packages to our library, which are used
      to read power grid data, modify them and perform simulations.

Contacts
--------

If you have in mind a use-case that would require some adjustments of the present package,
feel free to contact us at laurentpagnier@math.arizona.edu or balthazar.donon@uliege.be.

.. note::

    This project is currently under active development.

Contents
--------

.. toctree::
    :maxdepth: 1

    h2mg/index
    data_formalism
    dataset/index
    normalization
    neural_networks/index
    postprocessing
    interaction
    usecase/index
