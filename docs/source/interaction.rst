.. _backend:

Power Systems Backend
=====================

.. module:: ml4ps
.. module:: ml4ps.backend.interface
.. module:: ml4ps.backend.pandapower
.. module:: ml4ps.backend.pypowsybl

The power system backend is an essential part of this library. It allows to rely on power system packages (such as
`pandapower <http://www.pandapower.org>`_ and `pypowsybl <https://www.powsybl.org>`_) for the loading of power grid
files, the updating of power grid features, the power flow simulations, and the extraction of power grid features.

Backend interface
-----------------

We have defined a common backend interface through the abstract base class
:class:`ml4ps.backend.interface.AbstractBackend`.

.. autoclass:: ml4ps.backend.interface.AbstractBackend
    :members:

Discrepancies between backends
------------------------------

Every power grid backend has its own naming conventions and electro-technical models. To give a concrete example,
transformers are not modelled identically in `pandapower <http://www.pandapower.org>`_ and
`pypowsybl <https://www.powsybl.org>`_. As a consequence, they are not defined by the same features from one
package to the other. Moreover, certain advanced features are only available in certain power grid packages.

Current implementations
-----------------------

For now, only `pandapower <http://www.pandapower.org>`_ and `pypowsybl <https://www.powsybl.org>`_ have compatible
backend implementations in **ml4ps**. They can be accessed as follows :

.. code-block:: pycon

    import ml4ps as mp
    pandapowerbackend = mp.PandaPowerBackend()
    pypowsyblbackend = mp.PyPowSyblBackend()
