Interacting with Power Grids
============================

Le backend power systems est une composante essentielle de ce package. Comme le travail de modélisation
et de simulation a déjà été fait avec beaucoup de brio....
De plus, nous souhaitons proposer des outils qui n'ont pas vocation à être un nouveau package de simulation
de réseau, mais plutôt à assurer la jonction entre ces outils et les outils deep learning.
Pour ces raisons, nous avons défini une interface qui vient réutiliser quelques fonctions élémentaires que
l'on attend d'un package de power systems, et

Ce que nous appelons ici un backend a plusieurs utilisations

Reading power systems data
--------------------------


Une première utilisation du backend consiste en la lecture des fichiers power systems.
En fonction du backend, différentes extensions de fichiers sont lisibles.
Les extensions lisibles par un backend sont listées dans l'attribut `valid_extensions`.

.. code-block:: pycon

    >>> import ml4ps as mp
    >>> backend = mp.PandaPowerBackend()
    >>> backend.valid_extensions
    (".json", ".pkl")

Power Grid Modelling
--------------------

De plus, chaque package fait des choix de nomenclature et de modélisation différents. Pour donner un exemple concret,
les transformateurs ne sont pas modélisés par le même schéma électrique dans pandapower et dans pypowsybl, ce
qui a pour conséquence que chaque package va définir des features différentes pour chaque classe d'objet.
La richesse de modélisation étant variable d'un backend à l'autre, il est tout à fait possible qu'une même classe
d'objets soit représentée avec plus de features dans un backend que dans l'autre.
De même, tous les backend ne modélisent pas exactement les mêmes classes d'objets.

On peut trouver dans les attributs

.. note::

    Le choix du backend est extrêmement important, et doit se faire en ayant bien à l'esprit le type de problème
    que l'on souhaite résoudre.




Defining a new backend
----------------------

We have defined a common backend interface through the abstract base class
:class:`ml4ps.backend.interface.AbstractBackend`.
It requires a certain amount of elementary attributes and methods, and provides higher level methods based
upon the latter.

It is required to define the following attributes (see :class:`ml4ps.backend.pandapower.PandaPowerBackend`
for a concrete example) :

    - :attr:`ml4ps.backend.interface.valid_extensions`, a tuple of strings of all extensions
      that can be read by the package ;
    - :attr:`ml4ps.backend.interface.valid_features`, a dictionary of lists of strings,
      whose keys correspond to the object classes, and where values are the lists of feature names for each class ;
    - :attr:`ml4ps.backend.interface.valid_addresses`, a dictionary of lists of strings,
      whose keys correspond to the object classes, and where values are the lists of address names for each class ;

It is also required to override the following methods :

    - :meth:`ml4ps.backend.interface.load_network`, which loads a single instance of power grid ;
    - :meth:`ml4ps.backend.interface.set_feature_network`, which sets values of a single network according
      to values provided in a dictiona. This is useful when one wants to apply the output of a neural network
      to actual power instances ;
    - :meth:`ml4ps.backend.interface.run_network`, which runs a power flow simulation using the solver
      implemented in the backend ;
    - :meth:`ml4ps.backend.interface.get_feature_network`, which extracts features from a single power grid ;
    - :meth:`ml4ps.backend.interface.get_address_network`, which extracts addresses from a single power grid.

Interacting with power grids
----------------------------

The elementary operations that are required by the interface should only operate on single instances of power
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
