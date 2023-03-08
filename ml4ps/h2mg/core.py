from typing import Any, Callable, Dict, Iterator
from jax.tree_util import register_pytree_node_class, tree_map


# TODO : utiliser tree_map
# TODO : utiliser tree_structure pour vérifier que les structures sont les mêmes
# TODO : permettre d'agréger les features d'une même classe.
# TODO : vérifier que toutes les dims d'une même classe sont identiques.
# TODO : shallow repr cache pas mal de choses
# TODO : rendre le constructeur plus clair, avec plusieurs signatures possibles.

# TODO : changer le constructeur pour qu'on puisse créer un H2MG vide plus facilement, ou avec des signatures différentes

# TODO : garder à l'esprit qu'un H2MG peut aussi contenir des fonctions plutôt que des array
# TODO : si on initialise les données avec des listes ou des numpy array, il faut tout transformer en jax.array
# TODO :     ça permettra d'assurer la compatibilité des fonctions shap.
# TODO : que faire d'un H2MG qui contient des fonctions plutôt que des array ? Y a t il un attribut shape ?

# TODO : mettre un check pour s'assurer qu'il n'y a pas une addresse quelque part qui soit supérieure au max de
# TODO :     all_addresses.

# TODO : mettre un check sur la profondeur des différents dictionnaires
# TODO :     à voir, parce qu'on peut avoir des profondeurs variables.

# TODO : mettre un check pour s'assurer qu'il n'y a pas de classe locale qui s'appelle 'global'

# TODO : merge features : on rajoute une dimension si besoin sur chacune des features, et puis concatenate.
# TODO :     ça fait sauter un niveau au H2MG, donc il faut voir si c'est ok pour nous ou si on veut créer une sous classe
# TODO : en même temps, on aimerait bien pouvoir garder H2MG.local_features et global_features.

# TODO : Check list :
# TODO : 1/ Vérifier qu'il n'y a pas une clé inattendue


LOCAL_FEATURES = "local_features"
GLOBAL_FEATURES = "global_features"
LOCAL_ADDRESSES = "local_addresses"
ALL_ADDRESSES = "all_addresses"


@register_pytree_node_class
class H2MG(dict):
    """Hyper Heterogeneous Multi Graph (H2MG)."""

    def __init__(self, data_dict=None, check_data=True, **kwargs):
        data = data_dict if data_dict else kwargs
        if data:
            super().__init__({
                LOCAL_FEATURES: data.get(LOCAL_FEATURES, {}),
                GLOBAL_FEATURES: data.get(GLOBAL_FEATURES, {}),
                LOCAL_ADDRESSES: data.get(LOCAL_ADDRESSES, {}),
                ALL_ADDRESSES: data.get(ALL_ADDRESSES, [])
            })
            if check_data:
                self._check_data(data)
        else:
            super().__init__({LOCAL_FEATURES: {}, GLOBAL_FEATURES: {}, LOCAL_ADDRESSES: {}, ALL_ADDRESSES: []})

    def tree_flatten(self):
        children = self.values()
        aux = self.keys()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return H2MG({k: f for k, f in zip(aux_data, children)})

    def __add__(self, other) -> 'H2MG':
        """Element-wise addition of two H2MGs with same structure, or of one H2MG with a scalar quantity."""
        if isinstance(other, H2MG):
            return tree_map(lambda a, b: a + b, self, other)
        else:
            return tree_map(lambda a: a + other, self)

    def __radd__(self, other) -> 'H2MG':
        return self.__add__(other)

    def __sub__(self, other) -> 'H2MG':
        """Element-wise subtraction of two H2MGs with same structure, or of one H2MG with a scalar quantity."""
        if isinstance(other, H2MG):
            return tree_map(lambda a, b: a - b, self, other)
        else:
            return tree_map(lambda a: a - other, self)

    def __rsub__(self, other) -> 'H2MG':
        return - self.__sub__(other)

    def __mul__(self, other) -> 'H2MG':
        """Element-wise multiplication of two H2MGs with same structure, or of one H2MG with a scalar quantity."""
        if isinstance(other, H2MG):
            return tree_map(lambda a, b: a * b, self, other)
        else:
            return tree_map(lambda a: a * other, self)

    def __rmul__(self, other) -> 'H2MG':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'H2MG':
        """Element-wise division of two H2MGs with same structure, or of one H2MG with a scalar quantity."""
        if isinstance(other, H2MG):
            return tree_map(lambda a, b: a / b, self, other)
        else:
            return tree_map(lambda a: a / other, self)

    def __rtruediv__(self, other) -> 'H2MG':
        if isinstance(other, H2MG):
            return tree_map(lambda a, b: b / a, self, other)
        else:
            return tree_map(lambda a: other / a, self)

    def __pow__(self, exponent) -> 'H2MG':
        """Element-wise exponentiation of two H2MG with same structure, or of one H2MG with a scalar quantity."""
        if isinstance(exponent, H2MG):
            return tree_map(lambda a, b: a ** b, self, exponent)
        else:
            return tree_map(lambda a: a ** exponent, self)

    def __rpow__(self, base) -> 'H2MG':
        if isinstance(base, H2MG):
            return tree_map(lambda a, b: a ** b, base, self)
        else:
            return tree_map(lambda a: base ** a, self)

    def __neg__(self) -> 'H2MG':
        """Returns an H2MG with opposite features."""
        return self * (-1)

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return str(shallow_repr(self))

    def __getitem__(self, __key: Any) -> Any:
        if isinstance(__key, str):
            return super().__getitem__(__key)
        return tree_map(lambda a: a.__getitem__(__key), [self])

    @property
    def local_features(self) -> Dict:
        return self.get(LOCAL_FEATURES, {})

    @property
    def global_features(self) -> Dict:
        return self.get(GLOBAL_FEATURES, {})

    @property
    def local_addresses(self) -> Dict:
        return self.get(LOCAL_ADDRESSES, {})

    @property
    def all_addresses(self):
        return self.get(ALL_ADDRESSES, {})

    @local_features.setter
    def local_features(self, value):
        self[LOCAL_FEATURES] = value

    @global_features.setter
    def global_features(self, value):
        self[GLOBAL_FEATURES] = value

    @local_addresses.setter
    def local_addresses(self, value):
        self[LOCAL_ADDRESSES] = value

    @all_addresses.setter
    def all_addresses(self, value):
        self[ALL_ADDRESSES] = value

    @property
    def local_features_iterator(self) -> Iterator:
        for obj_name in self.get(LOCAL_FEATURES, {}):
            for feat_name, value in self[LOCAL_FEATURES][obj_name].items():
                yield LOCAL_FEATURES, obj_name, feat_name, value

    @property
    def local_addresses_iterator(self) -> Iterator:
        for obj_name in self.get(LOCAL_ADDRESSES, {}):
            for addr_name, value in self[LOCAL_ADDRESSES][obj_name].items():
                yield LOCAL_ADDRESSES, obj_name, addr_name, value

    @property
    def global_features_iterator(self) -> Iterator:
        for feat_name, value in self.get(GLOBAL_FEATURES, {}).items():
            yield GLOBAL_FEATURES, feat_name, value

    @property
    def all_addresses_iterator(self) -> Iterator:
        if ALL_ADDRESSES in self:
            yield ALL_ADDRESSES, self[ALL_ADDRESSES]

    @property
    def shape(self) -> 'H2MG':
        return tree_map(lambda a: a.shape, self)

    @property
    def size(self) -> 'H2MG':
        return tree_map(lambda a: a.size, self)

    @property
    def class_count(self) -> 'H2MG':
        shape_0 = self.shape[0]
        local_features_count = {k: max(max([v for v in d.values()])) for k, d in shape_0.local_features.items()}
        local_addresses_count = {k: max(max([v for v in d.values()])) for k, d in shape_0.local_features.items()}
        local_count = {k: max(local_features_count[k], local_addresses_count[k]) for k in local_features_count | local_addresses_count}
        global_count = max(max([d for d in shape_0.global_features.values()]))
        return local_count | {'global': global_count}

    def update(self, other) -> 'H2MG':
        self.global_features = self.global_features | other.global_features
        self.local_features = {k: self.local_features.get(k, {}) | other.local_features.get(k, {})
            for k in self.local_features | other.local_features}
        self.local_addresses = {k: self.local_addresses.get(k, {}) | other.local_addresses.get(k, {})
            for k in self.local_addresses | other.local_addresses}
        self.all_addresses = self.all_addresses if self.all_addresses else other.all_addresses

    def apply(self, fn: Callable | 'H2MG') -> 'H2MG':
        """Apply a single scalar function to all features, or a H2MG of functions with one function per tree_leaf. """
        if isinstance(fn, H2MG):
            return tree_map(lambda f, x: f(x), fn, self)
        else:
            return tree_map(fn, self)

    def plot(self):
        raise NotImplementedError

    def shallow_repr(self) -> Dict[str, Any]:
        return shallow_repr(self)

    def extract_like(self, other: 'H2MG') -> 'H2MG':
        """Extracts the features """
        r = H2MG()
        r.local_features = {k: {f: self.local_features[k][f] for f in d} for k, d in other.local_features.items()}
        r.local_addresses = {k: {f: self.local_addresses[k][f] for f in d} for k, d in other.local_addresses.items()}
        r.global_features = {k: self.global_features[k] for k in other.global_features}
        r.all_addresses = self.all_addresses
        return r

    def _check_data(self, data):
        if isinstance(data, list):
            return self._check_data(dict(data))
        valid_keys = [LOCAL_FEATURES, LOCAL_ADDRESSES, GLOBAL_FEATURES, ALL_ADDRESSES]
        if not set(data.keys()).issubset(set(valid_keys)):
            raise ValueError(f"Unknown keys in data: {data.keys()}, expected in {valid_keys}")


def shallow_repr(h2mg, local_features: bool = True, global_features: bool = True, local_addresses: bool = False,
                 all_addresses: bool = False) -> Dict[str, Any]:
    results = {}
    if local_features:
        for key, obj_name, feat_name, value in h2mg.local_features_iterator:
            results[obj_name + "_" + feat_name] = value
    if global_features:
        for key, feat_name, value in h2mg.global_features_iterator:
            results[feat_name] = value
    if local_addresses:
        for key, obj_name, feat_name, value in h2mg.local_addresses_iterator:
            results[obj_name + "_" + feat_name] = value
    if all_addresses:
        for key, value in h2mg.all_addresses_iterator:
            results[key] = value
    return results
