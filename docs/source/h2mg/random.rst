.. _random_h2mg:

Random H2MGs
============

Our package being aimed at training stochastic policies to control power grids, we propose
several implementations of standard distributions.
When possible, we propose a sampling function, as well as a function that computes the
log-probability of an `h2mg`_ (differentiable w.r.t the parameters of the distribution).

Contents
--------

.. module:: ml4ps.h2mg.random
.. autofunction:: uniform_like
.. autofunction:: normal_like
.. autofunction:: normal_logprob
.. autofunction:: categorical
.. autofunction:: categorical_logprob
.. autofunction:: categorical_per_feature
.. autofunction:: categorical_per_feature_logprob
