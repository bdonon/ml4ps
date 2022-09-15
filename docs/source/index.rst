ml4ps
=====

**ml4ps** (Machine Learning for Power Systems) is a Python library to facilitate the application of
Machine Learning techniques to Power Systems.

Nous souhaitons ainsi permettre aux gens d'entraîner facilement des modèles de réseaux de neurones
pour différentes tâches :
    - approximation de simulateur ;
    - approximation de solveur AC OPF ;
    - contrôle de certaines variables,
    - etc.

Nous fournissons les briques permettant d'atteindre ces buts, mais c'est à vous de définir
proprement les données et le problème que vous souhaitez résoudre. Des exemples d'applications
de nos outils sont disponibles dans la section usecase.

    - un formalisme de données qui respecte la structure de graph des données
    - un outil pour se brancher sur des bases de données de réseaux électriques, et itérer dessus
(basé sur torch)
    - un outil pour normaliser les données en respectant la structure ;
    - des implémentations de réseaux de neurones : un fully connected tout simple, et un h2mgnn
qui permet de respecter la structure des données;
    - des fonctions élémentaires pour postprocess
    - une interface pour possiblement brancher d'autres packages qui permettent de travailler sur
des réseaux, et qui peut permettre d'aller interagir avec les modèles de réseau.

Nous souhaitons ainsi faciliter l'application de méthodes avancées de machine learning pour l'opération
des réseaux électriques.

Nous proposons ainsi des outils pour apprendre depuis des données power systems sans aucun travail
lourd et sujet à erreur.


that want to apply Machine Leaning methods to Power Systems data.

It provides a way of iterating over power grid datasets, normalizing them,
feeding them into neural network architectures.

.. note::

    This project is under active development.

Contents
--------

.. toctree::
    :maxdepth: 1

    data_formalism
    usecase/index
    backend
    dataset
    normalization
    neural_networks/index
    postprocessing
    interaction
