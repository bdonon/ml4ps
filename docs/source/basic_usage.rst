Basic Usage
===========

Let us consider that we want to train a neural network to imitate the output of an AC power flow simulator.

Explanation
-----------

We have a dataset, from which we sample data, etc.

Usage
-----

.. code-block:: console

    x_norm = normalizer(x)
    y = neural_network(x_norm)
    y_pp = postprocessor(y)


