.. _data-formalism:

Power Grid Data Structure
=========================

In this section, we detail the data formalism used throughout the package.

Limits of vectorization
-----------------------

Early work on the application of machine learning to power systems basically relied on a pre-processing
step which amounted to vectorizing the power grid data. This was required because most machine learning
techniques were designed to work on vector data of constant sizes.

By doing so, people made the implicit assumption that the grid structure was the same in every sample
of the dataset :

    - Same amount of buses, generators, loads, lines, etc. throughout the dataset ;
    - Same object naming and ordering ;
    - Same interconnection patterns.

Unfortunately, none of those assumption actually hold in real-life power grid data.

Limits of standard graphs
-------------------------

In order to account for the above-mentioned limitations, recent work focused on the use of Graph Neural Networks
which allowed to work with graph data. This allowed to work with data that have a varying amount of objects,
that may be ordered differently from one sample to the other, and that can have varying interconnection patterns.

Still, this required to represent power grids as standard graphs, essentially made of vertices and edges.
While this is far better than vectorizing data, it still requires a preprocessing step to transform the
power grid into a graph composed of vertices and edges.

Now, let's say that you want to be able to predict action that should be applied to shunts.
Since shunts are only connected to one bus, you include the shunt in the node features.
Imagine that you have two shunts connected to the same bus : you have to merge them into a single
virtual object to fit your graph formalism. But controlling two distinct shunts is not equivalent to
controlling a twice as large shunt. By making this modelling choice, you are actually hindering your
data.

Hyper-Heterogeneous Multi Graphs
--------------------------------

Still, it is possible to represent power grid in a more natural way, which requires no preprocessing
that degrades the data structure.

Instead of considering vertices and edges, we are now considering hyper-edges and addresses.
Hyper-edges are a generalization of edges : they can be connected to a various amount of addresses.
In our formalism, only hyper-edges bear features, while addresses only serve as interconnection
ports between hyper-edges.

A power grid is composed of multiple classes of hyper-edges : buses, lines, loads, etc.
All hyper-edges that belong to the same class have :

    - the same order (i.e. are connected to the same amount of addresses). For instance all transmission
      lines are connected to exactly two addresses, generators are all connected to a single address, etc.
    - the same features. For instance, all loads have a feature that define their active power consumption,
      their reactive power consumption, etc.

In addition, we allow multiple objects of the same class to have the same address. You may now consider
multiple generators connected to the same bus, or multiple transmission lines.

To sum things up, we are dealing with Hyper-Heterogeneous Multi Graphs (H2MG) :

    - Hyper-graphs: Graphs that have hyper-edges, which can be connected to any number of vertices.
    - Heterogeneous graphs:Graphs that are made of multiple classes of objects.
    - Multi-graphs: Graphs that allow multiple objects to have the same addresses.

Mathematical formalism
----------------------

Hyper-edges and classes
_______________________

Let :math:`n \in \mathbb{N}`, and :math:`\mathcal{C}` be the set of considered classes.
We denote by :math:`\mathcal{E}^c` the set of hyper-edges of class :math:`c \in \mathcal{C}`.
All such hyper-edges are connected to the same amount of vertices through their ordered
ports :math:`\mathcal{O}^c`.
Thus, :math:`\mathcal{E}^c \subseteq [n]^{|\mathcal{O}^c|}`.
Classes such that :math:`|\mathcal{O}^c| = 1` represent objects that are located at exactly one vertex
(such as generators or loads).
Classes such that :math:`|\mathcal{O}^c| = 2` represent objects that are located at exactly two vertices
(such as transmission lines or transformers in power grids).

Multi objects
_____________

Let :math:`c \in \mathcal{C}` and :math:`e \in \mathcal{E}^c`.
We denote by :math:`\mathcal{M}^c_e` the set of objects of class :math:`c` that lie on hyper-edge :math:`e`.
Those objects may bear different feature vectors, and cannot be simply aggregated into an
equivalent object.

Structure
_________

A H2MG is composed of a structure that defines the interconnection patterns of objects,
and some features that are attached to each object.
We denote the structure of a H2MG :math:`x` as :math:`(n, \mathcal{C}, \mathcal{E}, \mathcal{M})`,
where :math:`\mathcal{E} = (\mathcal{E}^c)_{c \in \mathcal{C}}` and
:math:`\mathcal{M} = ((\mathcal{M}_e^c)_{e \in \mathcal{E}^c})_{c \in \mathcal{C}}`.

Moreover, we use the following simplifying notation:

.. math::

    \mathcal{G}_x = \{(c,e,m) | c \in \mathcal{C}, e \in \mathcal{E}^c, m \in \mathcal{M}_e^c \}

Let :math:`i \in [n]`.
We call hyper-edge neighborhood of a vertex the set of hyper-edges that are connected to it.

.. math::

    \mathcal{N}_x(i) = \{(c,e,m,o) | (c,e,m) \in \mathcal{G}_x, o \in \mathcal{O}^c, e_o=i\}

One may observe that this set returns the class, the hyper-edge, the multi-object id
and the port through which each object is connected to i.

Features
________

Contrarily to standard graphs, H2MGs exclusively bear features at hyper-edges:
vertices only play the role of addresses to which hyper-edges can be connected.
In that sense, vertices should be seen as an interface between hyper-edges.
The corresponding graph data can still be written as :math:`(x, y)` where :math:`x`
is the input and :math:`y` the output.

.. math::

    x = (x_{e,m}^c)_{(c,e,m) \in \mathcal{G}_x} \\
    y = (y_{e,m}^c)_{(c,e,m) \in \mathcal{G}_x}

.. image:: figures/h2mg.png
    :width: 800

    Power grid instance and its conversions into a standard graph, and into a H2MG.
    Standard graphs require to aggregate together vertex-like objects on the one hand
    and edge-like objects on the other hand.
    Meanwhile, H2MG allow to seamlessly represent power grids, without any information loss.
    In this example :math:`\mathcal{C} = \{\text{generator}, \text{load}, \text{line}, \text{transformer}\}`.
    Input features are in the following dimensions:
    :math:`d^{\text{gen},x} = 1`, :math:`d^{\text{load},x} = 2`,
    :math:`d^{\text{line},x} = 2`, :math:`d^{\text{transfo},x} = 3`.
    Lines and transformers are of order 2, while generators and loads are of order 1.
    For the sake of readability, only input features are considered.


Compatible neural network architecture
______________________________________

A special type of graph neural network has been developed jointly with this data formalism.
See :ref:`h2mgnode` for more details.

Numerical Representation
------------------------

Now that the mathematical definition of H2MGs has been introduced, we may now proceed to show are
they are implemented in our library.
Let us consider the power grid shown in the figure below.

.. image:: figures/simple_power_grid.png
    :width: 800

It is composed of multiple classes of hyper-edges : `bus`, `gen`, `load`, `line`
and `trafo`. The first three of them are of order 1 (i.e. are only connected
to one address), while the last two are of order 2 (i.e. are connected to two
addresses).

Addresses (that define the interconnection patterns of the graph) can be found in an object called `a`.
It is a nested dictionary :
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
