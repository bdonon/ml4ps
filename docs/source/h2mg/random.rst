.. _random_h2mg:

Random H2MGs
============

Our package being aimed at training stochastic policies to control power grids, we propose
several implementations of standard distributions.
When possible, we propose a sampling function, as well as a function that computes the
log-probability of an :ref:`H2MG <h2mg>` (differentiable w.r.t the parameters of the distribution).

Contents
--------



.. currentmodule:: ml4ps.h2mg.random

.. automodule:: ml4ps.h2mg.random

.. autosummary::
  :toctree: _autosummary

    categorical
    categorical_logprob
    categorical_per_feature