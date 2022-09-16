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


Interacting with power systems simulators
-----------------------------------------

Ces grandeurs devront ensuite pouvoir être utilisées pour remplacer les champs
correspondant dans les instances de réseau électrique, de sorte à pouvoir ensuite faire
appel .....

En gros on a 3 méthodes principales qui permettent d'interagir avec le backend :

    - set : applique les valeurs fournies aux instances de réseau.
    - run : fait tourner une simulation ac power flow
    - get : récupère des grandeurs issues du calcul de power flow

Bien entendu, cette dernière fonction ne permet que de récupérer des grandeurs qu'il
faudra ensuite combiner pour obtenir par exemple une fonction de coût.

Petit exemple d'interaction avec le backend

En fonction du problème, on utilisera l'une ou l'autre de ces méthodes. Dans le cas
où on cherche à apprendre la sortie du power flow, on pourra simplement utiliser run
et get.
Dans le cas où on cherche à entraîner un réseau de neurones à résoudre un opf de façon
non supervisée, on pourra utiliser set run et get.
C'est à l'utilisateur de définir de quelles méthodes il a besoin de la façon dont il
combine ensuite les grandeurs obtenues.

Changer le backend
------------------

We have defined a common backend interface through the abstract base class
:py:class:`~ml4ps.backend.interface.AbstractBackend`.
It requires a certain amount of elementary attributes and methods, and provides higher level methods based
upon the latter.
It is required to define the following attributes (see :py:class:`~ml4ps.backend.pandapower.PandaPowerBackend`
for a concrete example).

    - `valid_extensions`, a tuple of strings of all extensions that can be read by the package ;
    - `valid_features`, a dictionary of lists of strings, whose keys correspond to the object classes,
      and where values are the lists of feature names for each class ;
    - `valid_addresses`, a dictionary of lists of strings, whose keys correspond to the object classes,
      and where values are the lists of address names for each class ;


