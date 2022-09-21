.. _data-formalism:

Power Grid Data Structure
=========================



Numerical Representation
------------------------

Un des objectifs de ce projet est de représenter les données d'une façon qui
permette d'exprimer leur structure de façon explicite.
Par conséquent, nous avons choisi de distinguer dans les données ce qui relève
des addresses, et ce qui relève des features :

    a et x

On va donner un petit exemple :

For instance, let us consider the power grid shown in the figure below.

.. image:: figures/simple_power_grid.png
  :width: 800

It is composed of multiple classes of objects : `bus`, `gen`, `load`, `line`
and `trafo`. The first three of them are of order 1 (i.e. are only connected
to one address), while the last two are of order 2 (i.e. are connected to two
addresses).

The complete structure of the power grid is defined in `a`. It is a nested dictionary :
upper level keys correspond to the object class, while the lower level key
corresponds to the name of the address. For instance, transformers should be able
to differentiate between their `from` address and their `to` address.

.. code-block:: pycon

    >>> a['bus']
    {'name': [0, 1, 2, 3]}
    >>> a['load']
    {'bus': [2, 2, 3]}
    >>> a['gen']
    {'bus': [0, 3]}
    >>> a['line']
    {'from': [0, 2], 'to': [1, 3]}
    >>> a['trafo']
    {'from': [1, 1], 'to': [2, 3]}

On the other hand, features are stored in `x`. Once again, it is a nested dictionary,
where the upper level keys correspond to the various object classes, and the lower level
keys correspond to the different feature names of each class.

.. code-block:: pycon

    >>> x['load']
    {'p_mw': [12.3, 45.6, 78.9], 'q_mvar': [1.23, 4.56, 7.89]}
    >>> x['line']
    {'r': [0.01, 0.02], 'x': [0.03, 0.04], 'h': [0.05, 0.06]

Indeed, those values were made up, and `x` is very likely to contain way
more features, but this gives an idea of the data structure.

Il faut dire aussi que les grandeurs renvoyées par nos réseaux de neurones partageront
la même structure de nested dictionary.
