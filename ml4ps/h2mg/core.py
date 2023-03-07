from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List
from functools import partial
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class H2MG(dict):
    """ Hyper Heterogeneous Multi Graph (H2MG).

    Data formalism that can seamlessly represent power grid, without any preprocessing or aggregation.

    """

    def __init__(self, data, check_data=True):
        if check_data:
            self._check_data(data)
        super().__init__(data)

    def tree_flatten(self):
        children = self.values()
        aux = self.keys()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return H2MG({k: f for k, f in zip(aux_data, children)})

    @property
    def local_features(self) -> Dict:
        return self.get(H2MGCategories.LOCAL_FEATURES.value, {})

    @property
    def global_features(self) -> Dict:
        return self.get(H2MGCategories.GLOBAL_FEATURES.value, {})

    @property
    def local_addresses(self) -> Dict:
        return self.get(H2MGCategories.LOCAL_ADDRESSES.value, {})

    @property
    def all_addresses(self):
        return self.get(H2MGCategories.ALL_ADDRESSES.value, {})

    @local_features.setter
    def local_features(self, value):
        self[H2MGCategories.LOCAL_FEATURES.value] = value

    @global_features.setter
    def global_features(self, value):
        self[H2MGCategories.GLOBAL_FEATURES.value] = value

    @local_addresses.setter
    def local_addresses(self, value):
        self[H2MGCategories.LOCAL_ADDRESSES.value] = value

    @all_addresses.setter
    def all_addresses(self, value):
        self[H2MGCategories.ALL_ADDRESSES.value] = value

    def __add__(self, other) -> 'H2MG':
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a + b, [self, other])
        else:
            return map_to_features(lambda a: a + other, [self])

    def __sub__(self, other) -> 'H2MG':
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a - b, [self, other])
        else:
            return map_to_features(lambda a: a - other, [self])

    def __mul__(self, other) -> 'H2MG':
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a * b, [self, other])
        else:
            return map_to_features(lambda a: a * other, [self])

    def __rmul__(self, other) -> 'H2MG':
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a * b, [self, other])
        else:
            return map_to_features(lambda a: a * other, [self])

    def __truediv__(self, other) -> 'H2MG':
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a / b, [self, other])
        else:
            return map_to_features(lambda a: a / other, [self])

    def __pow__(self, exponent: int) -> 'H2MG':
        if isinstance(exponent, H2MG):
            return map_to_features(lambda a, b: a / b, [self, exponent])
        return map_to_features(lambda a: a ** exponent, [self])

    def __neg__(self) -> 'H2MG':
        return self * (-1)

    def log(self) -> 'H2MG':
        return map_to_features(jnp.log, [self])

    def exp(self) -> 'H2MG':
        return map_to_features(jnp.exp, [self])

    def maximum(self, other):
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: jnp.maximum(a, b), [self, other])
        else:
            return map_to_features(lambda a: jnp.maximum(a, other), [self])

    def stack(self, other):
        return map_to_features(lambda a, b: jnp.stack([a, b], axis=-1), [self, other])

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return str(shallow_repr(self))

    def __getitem__(self, __key: Any) -> Any:
        if isinstance(__key, str):
            return super().__getitem__(__key)
        return map_to_features(lambda a: a.__getitem__(__key), [self])

    def apply(self, fn: Callable[[Dict], Dict]) -> 'H2MG':
        return map_to_features(fn, [self])

    def apply_h2mg_fn(self, h2mg_fn: Dict) -> 'H2MG':
        return h2mg_apply(h2mg_fn, self)

    def sum(self) -> float:
        return sum(features_iterator(self.apply(jnp.sum)))

    def nansum(self) -> float:
        return sum(features_iterator(self.apply(jnp.nansum)))

    def combine(self, other) -> 'H2MG':
        return combine_space(self, other)

    def plot(self):
        raise NotImplementedError

    def shallow_repr(self) -> Dict[str, Any]:
        return shallow_repr(self)

    @property
    def local_features_iterator(self) -> Iterator:
        return local_features_iterator(self)

    @property
    def local_addresses_iterator(self) -> Iterator:
        return local_addresses_iterator(self)

    @property
    def global_features_iterator(self) -> Iterator:
        return global_features_iterator(self)

    @property
    def all_addresses_iterator(self) -> Iterator:
        return all_addresses_iterator(self)

    @property
    def features(self) -> Iterator:
        return features_iterator(self)

    @property
    def shape(self) -> 'H2MG':
        return map_to_features(lambda a: a.shape, [self])

    @property
    def size(self) -> 'H2MG':
        return map_to_features(lambda a: a.size, [self])

    # def __len__(self) -> int:
    #     return len(self.features)

    def extract_like(self, other: 'H2MG') -> 'H2MG':
        build_dict = dict
        output_h2mg = dict()
        for local_key, obj_name, feat_name, value in other.local_features_iterator:
            if local_key not in output_h2mg:
                output_h2mg[local_key] = build_dict()
            if obj_name not in output_h2mg[local_key]:
                output_h2mg[local_key][obj_name] = build_dict()
            output_h2mg[local_key][obj_name][feat_name] = self[local_key][obj_name][feat_name]

        for global_key, feat_name, value in other.global_features_iterator:
            if global_key not in output_h2mg:
                output_h2mg[global_key] = build_dict()
            output_h2mg[global_key][feat_name] = self[global_key][feat_name]

        for local_key, obj_name, addr_name, value in other.local_addresses_iterator:
            if local_key not in output_h2mg:
                output_h2mg[local_key] = build_dict()
            if obj_name not in output_h2mg[local_key]:
                output_h2mg[local_key][obj_name] = build_dict()
            output_h2mg[local_key][obj_name][addr_name] = self[local_key][obj_name][addr_name]

        for all_addr_key, value in other.all_addresses_iterator:
            output_h2mg[all_addr_key] = self[all_addr_key]
        return H2MG(output_h2mg)

    def flatten(self) -> jnp.ndarray:
        flat_dim = sum(v.size for v in self.features)
        flat_action = jnp.zeros(shape=flat_dim)
        i = 0
        for v in self.features:
            flat_action = flat_action.at[i:i + v.size].set(v.flatten())
            i += v.size
        return flat_action

    def unflatten_like(self, x) -> 'H2MG':
        res = empty_like(self)
        i = 0
        for key, obj_name, feat_name, value in self.local_features_iterator:
            res[key][obj_name][feat_name] = x[i:i + value.size].reshape(value.shape)
            i += value.size
        for key, feat_name, value in self.global_features_iterator:
            res[key][feat_name] = x[i:i + value.size].reshape(value.shape)
            i += value.size
        if i != x.size:
            raise ValueError(f"{i}!= {x.size}")
        return res

    def _check_data(self, data):
        if isinstance(data, list):
            return self._check_data(dict(data))
        valid_keys = [H2MGCategories.LOCAL_FEATURES.value, H2MGCategories.LOCAL_ADDRESSES.value,
            H2MGCategories.GLOBAL_FEATURES.value, H2MGCategories.ALL_ADDRESSES.value]
        if not set(data.keys()).issubset(set(valid_keys)):
            raise ValueError(f"Unknown keys in data: {data.keys()}, expected in {valid_keys}")

    @staticmethod
    def make(local_features, global_features, local_addresses, all_addresses):
        data = {
            H2MGCategories.LOCAL_FEATURES.value: local_features, H2MGCategories.GLOBAL_FEATURES.value: global_features,
            H2MGCategories.LOCAL_ADDRESSES.value: local_addresses, H2MGCategories.ALL_ADDRESSES.value: all_addresses}
        H2MG._check_data(data)
        return H2MG(data)


def combine_space(a, b):
    x = empty_h2mg()
    for local_key, obj_name, feat_name, value in local_features_iterator(a):
        x[local_key][obj_name][feat_name] = value
    for local_key, obj_name, feat_name, value in local_features_iterator(b):
        x[local_key][obj_name][feat_name] = value

    for global_key, feat_name, value in global_features_iterator(a):
        x[global_key][feat_name] = value
    for global_key, feat_name, value in global_features_iterator(b):
        x[global_key][feat_name] = value

    for local_key, obj_name, addr_name, value in local_addresses_iterator(a):
        x[local_key][obj_name][addr_name] = value
    for local_key, obj_name, addr_name, value in local_addresses_iterator(b):
        x[local_key][obj_name][addr_name] = value

    for all_addr_key, value in all_addresses_iterator(a):
        x[all_addr_key] = value
    return H2MG(x)


class H2MGCategories(Enum):

    LOCAL_FEATURES = "local_features"

    GLOBAL_FEATURES = "global_features"

    LOCAL_ADDRESSES = "local_addresses"

    ALL_ADDRESSES = "all_addresses"


def h2mg(local_features, global_features, local_addresses, all_addresses):
    return {
        H2MGCategories.LOCAL_FEATURES.value: local_features, H2MGCategories.GLOBAL_FEATURES.value: global_features,
        H2MGCategories.LOCAL_ADDRESSES.value: local_addresses, H2MGCategories.ALL_ADDRESSES.value: all_addresses}


def local_features(h2mg: Dict) -> Dict:
    return h2mg.get(H2MGCategories.LOCAL_FEATURES.value, {})


def global_features(h2mg):
    return h2mg.get(H2MGCategories.GLOBAL_FEATURES.value, {})


def local_addresses(h2mg):
    return h2mg.get(H2MGCategories.LOCAL_ADDRESSES.value, {})


def all_addresses(h2mg):
    return h2mg.get(H2MGCategories.ALL_ADDRESSES.value, [])


def local_feature_names_iterator(feature_names):
    local_key = H2MGCategories.LOCAL_FEATURES.value
    if local_key in feature_names:
        for obj_name, feat_names_list in feature_names[local_key].items():
            for feat_name in feat_names_list:
                yield local_key, obj_name, feat_name


def global_feature_names_iterator(feature_names):
    global_key = H2MGCategories.GLOBAL_FEATURES.value
    if global_key in feature_names:
        for feat_name in feature_names[global_key]:
            yield global_key, feat_name


def local_features_iterator(h2mg) -> Iterator:
    local_key = H2MGCategories.LOCAL_FEATURES.value
    if h2mg.get(local_key, None) is not None:
        for obj_name in h2mg[local_key]:
            for feat_name, value in h2mg[local_key][obj_name].items():
                yield local_key, obj_name, feat_name, value


def global_features_iterator(h2mg) -> Iterator:
    global_key = H2MGCategories.GLOBAL_FEATURES.value
    if h2mg.get(global_key, None) is not None:
        for feat_name, value in h2mg[global_key].items():
            yield global_key, feat_name, value


def local_addresses_iterator(h2mg) -> Iterator:
    local_key = H2MGCategories.LOCAL_ADDRESSES.value
    if h2mg.get(local_key, None) is not None:
        for obj_name in h2mg[local_key]:
            for addr_name, value in h2mg[local_key][obj_name].items():
                yield local_key, obj_name, addr_name, value


def all_addresses_iterator(h2mg) -> Iterator:
    all_addr_key = H2MGCategories.ALL_ADDRESSES.value
    if h2mg.get(all_addr_key, None) is not None:
        yield all_addr_key, h2mg[all_addr_key]


def features_iterator(h2mg) -> Iterator:
    for _, _, _, value in local_features_iterator(h2mg):
        yield value
    for _, _, value in global_features_iterator(h2mg):
        yield value


def empty_h2mg():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(float)))


def h2mg_slicer(key, obj_name, feat_name):
    def slice(h2mg):
        if key == H2MGCategories.LOCAL_FEATURES.value:
            return h2mg[key][obj_name][feat_name]
        elif key == H2MGCategories.GLOBAL_FEATURES.value:
            return h2mg[key][feat_name]
        elif key == H2MGCategories.LOCAL_ADDRESSES.value:
            return h2mg[key][obj_name][feat_name]
        elif key == H2MGCategories.ALL_ADDRESSES.value:
            return h2mg[key]
        else:
            raise ValueError

    return slice


def compatible(h2mg, h2mg_other):
    if not (isinstance(h2mg, dict) and isinstance(h2mg_other, dict)):
        return False
    if set(h2mg.keys()) != set(h2mg_other.keys()):
        return False
    local_key = H2MGCategories.LOCAL_FEATURES.value
    if set(h2mg.get(local_key, [])) != set(h2mg_other.get(local_key, [])):
        return False
    for obj_name in h2mg.get(local_key, []):
        if set(h2mg[local_key].get(obj_name, [])) != set(h2mg_other[local_key].get(obj_name, [])):
            return False
    local_addr_key = H2MGCategories.LOCAL_ADDRESSES.value
    if set(h2mg.get(local_addr_key, [])) != set(h2mg_other.get(local_addr_key, [])):
        return False
    for feat_name in h2mg.get(local_addr_key, []):
        if set(h2mg[local_addr_key].get(feat_name, [])) != set(h2mg_other[local_addr_key].get(feat_name, [])):
            return False
    global_key = H2MGCategories.GLOBAL_FEATURES.value
    if set(h2mg.get(global_key, [])) != set(h2mg_other.get(global_key, [])):
        return False
    return True


def h2mg_apply(fns_h2mg: H2MG, target_h2mg: H2MG, local_features: bool = True, global_features: bool = True,
               local_addresses: bool = False, all_addresses: bool = False) -> H2MG:
    if local_features:
        for key, obj_name, feat_name, value in local_features_iterator(target_h2mg):
            fn = fns_h2mg.get(key, {}).get(obj_name, {}).get(feat_name, lambda x: x)
            target_h2mg[key][obj_name][feat_name] = fn(value)
    if global_features:
        for key, feat_name, value in global_features_iterator(target_h2mg):
            fn = fns_h2mg.get(key, {}).get(feat_name, lambda x: x)
            target_h2mg[key][feat_name] = fn(value)
    if local_addresses:
        for key, obj_name, feat_name, value in local_addresses_iterator(target_h2mg):
            fn = fns_h2mg.get(key, {}).get(obj_name, {}).get(feat_name, lambda x: x)
            target_h2mg[key][obj_name][feat_name] = fn(value)
    if all_addresses:
        for key, value in all_addresses_iterator(target_h2mg):
            fn = fns_h2mg.get(key, lambda x: x)
            target_h2mg[key] = fn(value)

    return H2MG(target_h2mg)


def all_compatible(*h2mgs):
    if len(h2mgs) < 2:
        return True
    else:
        return compatible(h2mgs[0], h2mgs[1]) and all_compatible(*h2mgs[1:])


def empty_like(h2mg):
    new_h2mg = {}
    for key, obj_name, feat_name, value in local_features_iterator(h2mg):
        if key not in new_h2mg:
            new_h2mg[key] = {}
        if obj_name not in new_h2mg[key]:
            new_h2mg[key][obj_name] = {}
        if feat_name not in new_h2mg[key][obj_name]:
            new_h2mg[key][obj_name][feat_name] = None  # jnp.empty_like(jnp.array(value))

    for key, feat_name, value in global_features_iterator(h2mg):
        if key not in new_h2mg:
            new_h2mg[key] = {}
        if feat_name not in new_h2mg[key]:
            new_h2mg[key][feat_name] = None  # jnp.empty_like(jnp.array(value))

    for key, obj_name, addr_name, value in local_addresses_iterator(h2mg):
        if key not in new_h2mg:
            new_h2mg[key] = {}
        if obj_name not in new_h2mg[key]:
            new_h2mg[key][obj_name] = {}
        if addr_name not in new_h2mg[key][obj_name]:
            new_h2mg[key][obj_name][addr_name] = value

    for key, value in all_addresses_iterator(h2mg):
        new_h2mg[key] = value

    return H2MG(new_h2mg)


def collate_h2mgs_features(h2mgs_list):
    def collate_arrays(*args):
        return jnp.array(list(args))

    return map_to_features(collate_arrays, h2mgs_list)


def h2mg_map(fn: Callable, args_h2mg: List = None, local_features: bool = True, global_features: bool = True,
             local_addresses: bool = False, all_addresses: bool = False, check_compat: bool = False) -> Dict:
    if not args_h2mg:
        raise ValueError
    if check_compat and not all_compatible(args_h2mg):
        raise ValueError
    results = empty_like(args_h2mg[0])
    if local_features:
        for key, obj_name, feat_name, value in local_features_iterator(results):
            results[key][obj_name][feat_name] = fn(*list(map(h2mg_slicer(key, obj_name, feat_name), args_h2mg)))
    if global_features:
        for key, feat_name, value in global_features_iterator(results):
            results[key][feat_name] = fn(*list(map(h2mg_slicer(key, None, feat_name), args_h2mg)))
    if local_addresses:
        for key, obj_name, feat_name, value in local_addresses_iterator(results):
            results[key][obj_name][feat_name] = fn(*list(map(h2mg_slicer(key, obj_name, feat_name), args_h2mg)))
    if all_addresses:
        for key, value in all_addresses_iterator(results):
            results[key] = fn(*list(map(h2mg_slicer(key, None, None), args_h2mg)))

    return H2MG(results)


def map_to_features(fn: Callable, args_h2mg: List[H2MG], check_compat=False) -> H2MG:
    if not isinstance(args_h2mg, list):
        raise ValueError
    return h2mg_map(fn=fn, args_h2mg=args_h2mg, check_compat=check_compat)


def map_to_all(fn: Callable, args_h2mg: List, check_compat=False) -> H2MG:
    return h2mg_map(fn, args_h2mg=args_h2mg, local_features=True, global_features=True, local_addresses=True,
                    all_addresses=True, check_compat=check_compat)


def collate_h2mgs(h2mgs_list):
    def collate_arrays(*args):
        return jnp.array(list(args))

    return map_to_all(collate_arrays, h2mgs_list)


def shallow_repr(h2mg, local_features: bool = True, global_features: bool = True, local_addresses: bool = False,
                 all_addresses: bool = False) -> Dict[str, Any]:
    results = {}
    if local_features:
        for key, obj_name, feat_name, value in local_features_iterator(h2mg):
            results[obj_name + "_" + feat_name] = value
    if global_features:
        for key, feat_name, value in global_features_iterator(h2mg):
            results[feat_name] = value
    if local_addresses:
        for key, obj_name, feat_name, value in local_addresses_iterator(h2mg):
            results[obj_name + "_" + feat_name] = value
    if all_addresses:
        for key, value in all_addresses_iterator(h2mg):
            results[key] = value
    return results



