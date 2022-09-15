ml4ps
=====

**ml4ps** (Machine Learning for Power Systems) is a Python library to facilitate the application of
Machine Learning to Power Systems, with a strong emphasis on respecting the data structure.

We provide building blocks that allow to plug deep learning models to power grid data, and to interact
with various power systems backends (pandapower and pypowsybl).
At the core of this package is the idea that
real-life power systems have a structure that varies through time (because of line disconnections,
unforeseen incidents, the building of new facilities). For this reason, we believe that developing
AI models that respect the graph structure of power grid data is critical to their application to
real-life data.

We thus provide a series of tools that were built with this concern in mind :

    - a data formalism that properly describes the actual structure of the data ;
    - a dataset class derived from the `PyTorch <https://pytorch.org/docs/stable/data.html#
      single-and-multi-process-data-loading>`_ data loading utility, which returns objects
      that exhaustively describe power grid instances ;
    - a normalizer class that maps ill-distributed features into a more appropriate range of values ;
    - an implementation of a graph neural ordinary differential equation model in
      `JAX <https://jax.readthedocs.io/en/latest/>`_ that structurally
      respects the data structure ;
    - a post-processing class that sends the neural network output into physically meaningful orders
      of magnitudes ;
    - an interface that allows to read data using different backend implementations (for now only
      pandapower and pypowsybl are available), setting power grid features using neural network outputs,
      running power flow simulations and getting features from the simulation results.

These tools were built with several key applications in mind, among which are :

    - the imitation of an AC power flow simulator using neural networks ;
    - the imitation of an AC optimal power flow solver using neural networks.

If you have in mind a use-case that would require some adjustments of the present package,
feel free to contact us at laurentpagnier@math.arizona.edu or balthazar.donon@uliege.be.

.. note::

    This project is currently under active development.

Contents
--------

.. toctree::
    :maxdepth: 1

    data_formalism
    backend
    dataset
    normalization
    neural_networks/index
    postprocessing
    interaction
    usecase/index
