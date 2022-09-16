Power Systems Backend
=====================

.. module:: ml4ps.backend.interface
.. module:: ml4ps.backend.pandapower
.. module:: ml4ps.backend.pypowsybl

The power system backend is an essential part of this library. It allows to rely on power system packages (such as
`pandapower <http://www.pandapower.org>`_ and `pypowsybl <https://www.powsybl.org>`_) for the loading of power grid
files, the updating of power grid features, the power flow simulations, and the extraction of power grid features.

Backend interface
-----------------


We have defined a common backend interface through the abstract base class
:class:`AbstractBackend`.
:class:`test`
It requires to override the following attributes (see :class:`PandaPowerBackend`
for a concrete example) :

    - :attr:`AbstractBackend.valid_extensions` : a tuple of strings of all extensions
      that can be read by the package.
    - :attr:`AbstractBackend.valid_feature_names` : a dictionary of lists of strings,
      whose keys correspond to the object classes, and where values are the lists of feature names for each class.
    - :attr:`AbstractBackend.valid_address_names` : a dictionary of lists of strings,
      whose keys correspond to the object classes, and where values are the lists of address names for each class.

It also requires to override the following methods :

    - :meth:`AbstractBackend.load_network` : loads a single instance of power grid. The
      implementation should be consistent with the valid extensions defined in
      :attr:`AbstractBackend.valid_extensions`.
    - :meth:`AbstractBackend.set_feature_network` : sets features of a power grid instance according
      to nested dictionary. This is useful when one wants to apply the output of a neural network
      to actual power instances. The provided features should match :attr:`ml4ps.backend.interface.valid_feature_names`.
    - :meth:`ml4ps.backend.interface.run_network` : runs a power flow simulation using the backend's solver.
    - :meth:`ml4ps.backend.interface.get_feature_network` : extracts feature values from a single power grid. It
      should be consistent with :attr:`ml4ps.backend.interface.valid_feature_names`.
    - :meth:`ml4ps.backend.interface.get_address_network` : extracts address values from a single power grid. It
      should be consistent with :attr:`ml4ps.backend.interface.valid_address_names`.

Interacting with power grids
----------------------------

The elementary operations that are required by the interface operate on single instances of power
grids. Then, those methods are converted into batch operations as follows :

    - :meth:`ml4ps.backend.interface.set_feature_batch`, which sets values of a batch of power grids.
      network according
      to values provided in a dictiona. This is useful when one wants to apply the output of a neural network
      to actual power instances ;
    - :meth:`ml4ps.backend.interface.run_batch`, which runs a power flow simulation using the solver
      implemented in the backend ;
    - :meth:`ml4ps.backend.interface.get_feature_batch`, which extracts features from a single power grid ;

Those three basic methods will serve to interact with batches of power grids, allowing to replace values by
the batch output of a neural network (for instance), then performing power flow simulations over the batch of
power grid instances, and finally retrieving some relevant features that result from these computations.

Discrepancies between backends
------------------------------

Every power grid backend has its own naming conventions and electro-technical models. To give a concrete example,
transformers are not modelled identically in `pandapower <http://www.pandapower.org>`_ and
`pypowsybl <https://www.powsybl.org>`_. As a consequence, they are not defined by the same features from one
package to the other. Moreover, certain advanced features are only available in certain power grid packages.

Current implementations
-----------------------

For now, only `pandapower <http://www.pandapower.org>`_ and `pypowsybl <https://www.powsybl.org>`_ have compatible
backends in the implementation of **ml4ps**. They can be accessed as follows :

.. code-block:: pycon

    import ml4ps as mp
    pandapowerbackend = mp.PandaPowerBackend()
    pypowsyblbackend = mp.PyPowSyblBackend()

Contents
--------

.. autoclass:: ml4ps.backend.interface.AbstractBackend
    :members:

.. autoclass:: ml4ps.backend.pandapower.PandaPowerBackend
    :members:
