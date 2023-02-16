from collections import defaultdict
from enum import Enum
from typing import Callable, Iterator, List, Dict, Any

import jax.numpy as jnp
import jax
from jax.tree_util import register_pytree_node_class

from gymnasium import spaces
from functools import partial



@register_pytree_node_class
class H2MG(dict):

    def __init__(self, data):
        super().__init__(data)
    
    def tree_flatten(self):
        children = self.values()
        aux = self.keys()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return H2MG({k:f for k, f in zip(aux_data, children)})
    
    @property
    def local_features(self):
        return self.get(H2MGCategories.LOCAL_FEATURES.value, {})
    
    @property
    def global_features(self):
        return self.get(H2MGCategories.GLOBAL_FEATURES.value, {})
    
    @property
    def local_addresses(self):
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
        self[H2MGCategories.LOCAL_ADDRESSES.value] = value
    
    def __add__(self, other):
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a + b, [self, other])
        else:
            return map_to_features(lambda a: a + other, [self])
    
    def __sub__(self, other):
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a - b, [self, other])
        else:
            return map_to_features(lambda a: a - other, [self])
    
    def __mul__(self, other):
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a * b, [self, other])
        else:
            return map_to_features(lambda a: a * other, [self])
    
    def __rmul__(self, other):
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a * b, [self, other])
        else:
            return map_to_features(lambda a: a * other, [self])
    
    def __truediv__(self, other):
        if isinstance(other, H2MG):
            return map_to_features(lambda a, b: a / b, [self, other])
        else:
            return map_to_features(lambda a: a / other, [self])
    
    def __pow__(self, exponent: int):
        if isinstance(exponent, H2MG):
            return map_to_features(lambda a, b: a / b, [self, exponent])
        return map_to_features(lambda a: a ** exponent, [self])
    
    def __neg__(self):
        return self*(-1)
    
    def log(self):
        return map_to_features(jnp.log, [self])
    
    def exp(self):
        return map_to_features(jnp.exp, [self])
        
    def __repr__(self) -> str:
        return super().__repr__()
    
    def __str__(self) -> str:
        return str(shallow_repr(self))
    
    def __getitem__(self, __key: Any) -> Any:
        if isinstance(__key, str):
            return super().__getitem__(__key)
        return map_to_features(lambda a: a.__getitem__(__key), [self])
    
    def apply(self, fn: Callable[[Dict], Dict]):
        return map_to_features(fn, [self])
    
    def apply_h2mg_fn(self, h2mg_fn: Dict):
        return h2mg_apply(h2mg_fn, self)
    
    def __iter__(self) -> Iterator:
        return features_iterator(self)
    
    def sum(self):
        return sum(features_iterator(self.apply(jnp.sum)))

    def nansum(self):
        return sum(features_iterator(self.apply(jnp.nansum)))
    
    def combine(self, other):
        return combine_space(self, other)
    
    def plot(self):
        raise NotImplementedError
    
    def shallow_repr(self):
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
    def shape(self):
        return map_to_features(lambda a: a.shape, [self])
    
    def flatten(self):
        flat_dim = sum(v.size for v in self.features)
        flat_action = jnp.zeros(shape=flat_dim)
        i=0
        for v in self.features:
            flat_action = flat_action.at[i:i+v.size].set(v.flatten())
            i+=v.size
        return flat_action

    def unflatten_like(self, x):
        res = empty_like(self)
        i = 0
        for key, obj_name, feat_name, value in self.local_features_iterator:
            res[key][obj_name][feat_name] = x[i:i+value.size].reshape(value.shape)
            i+=value.size
        for key, feat_name, value in self.global_features_iterator:
            res[key][feat_name] = x[i:i+value.size].reshape(value.shape)
            i+=value.size
        if i != x.size:
            raise ValueError(f"{i}!= {x.size}")
        return res
    

def combine_space(a, b):
    x = empty_h2mg()
    for local_key, obj_name, feat_name, value in local_features_iterator(a):
        x[local_key][obj_name][feat_name] = value
    for local_key, obj_name, feat_name, value in local_features_iterator(b):
        x[local_key][obj_name][feat_name] = value

    for global_key,  feat_name, value in global_features_iterator(a):
        x[global_key][feat_name] = value
    for global_key,  feat_name, value in global_features_iterator(b):
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
    return {H2MGCategories.LOCAL_FEATURES.value: local_features,
            H2MGCategories.GLOBAL_FEATURES.value: global_features,
            H2MGCategories.LOCAL_ADDRESSES.value: local_addresses,
            H2MGCategories.ALL_ADDRESSES.value: all_addresses}


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
            yield global_key,  feat_name, value


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
    for _,  _, value in global_features_iterator(h2mg):
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
            new_h2mg[key][obj_name][feat_name] = None #jnp.empty_like(jnp.array(value))

    for key, feat_name, value in global_features_iterator(h2mg):
        if key not in new_h2mg:
            new_h2mg[key] = {}
        if feat_name not in new_h2mg[key]:
            new_h2mg[key][feat_name] = None # jnp.empty_like(jnp.array(value))

    for key, obj_name, addr_name, value in local_addresses_iterator(h2mg):
        if key not in new_h2mg:
            new_h2mg[key] = {}
        if obj_name not in new_h2mg[key]:
            new_h2mg[key][obj_name] = {}
        if addr_name not in new_h2mg[key][obj_name]:
            new_h2mg[key][obj_name][addr_name] = None # value

    for key, value in all_addresses_iterator(h2mg):
        new_h2mg[key] = None #value

    return H2MG(new_h2mg)

def collate_h2mgs_features(h2mgs_list):
    def collate_arrays(*args):
        return jnp.array(list(args))
    return map_to_features(collate_arrays, h2mgs_list)

def h2mg_map(fn: Callable, args_h2mg: List=None, local_features: bool=True, global_features: bool=True, local_addresses: bool=False, all_addresses: bool=False, check_compat: bool=False, in_place=False) -> Dict:
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

def map_to_features(fn, args_h2mg, check_compat=False, in_place=False) -> H2MG:
    return h2mg_map(fn=fn, args_h2mg=args_h2mg, check_compat=check_compat, in_place=in_place)

def map_to_all(fn: Callable, args_h2mg: List, check_compat=False) -> H2MG:
    return h2mg_map(fn, args_h2mg=args_h2mg, local_features=True, global_features=True, local_addresses=True, all_addresses=True, check_compat=check_compat)

def collate_h2mgs(h2mgs_list):
    def collate_arrays(*args):
        return jnp.array(list(args))
    return map_to_all(collate_arrays, h2mgs_list)

def h2mg_apply(norm_fns_h2mg, target_h2mg):
    return map_to_features(lambda feature, norm_fn: norm_fn(feature), args_h2mg=[target_h2mg, norm_fns_h2mg])

def shallow_repr(h2mg, local_features: bool=True, global_features: bool=True, local_addresses: bool=False, all_addresses: bool=False):
    results = {}
    if local_features:
        for key, obj_name, feat_name, value in local_features_iterator(h2mg):
            results[obj_name+"_"+feat_name] = value
    if global_features:
        for key, feat_name, value in global_features_iterator(h2mg):
            results[feat_name] = value
    if local_addresses:
        for key, obj_name, feat_name, value in local_addresses_iterator(h2mg):
            results[obj_name+"_"+feat_name] = value
    if all_addresses:
        for key, value in all_addresses_iterator(h2mg):
            results[key] = value
    return results


def uniform_like(h2mg: H2MG, rng):
    pass

def normal_like(rng, h2mg: H2MG) -> H2MG:
    flat_h2mg = h2mg.flatten()
    x = jax.random.normal(rng, flat_h2mg.shape)
    return h2mg.unflatten_like(x)

def normal_logprob(x: H2MG, mu: H2MG, log_sigma: H2MG) -> float:
    return (- log_sigma - 0.5 * (-2 * log_sigma).exp() * (jax.lax.stop_gradient(x) - mu)**2).nansum()

def categorical(rng, logits: H2MG) -> H2MG:
    flat_logits = logits.flatten()
    idx = jax.random.categorical(key=rng, logits=flat_logits)
    flat_res_action = jnp.zeros_like(flat_logits)
    flat_res_action = flat_res_action.at[idx].set(1)
    res_action = logits.unflatten_like(flat_res_action)
    return res_action
    # res_action = self.onehot_to_action(res_action)

def categorical_logprob(x_onehot: H2MG, logits: H2MG) -> float:
    flat_onehot = x_onehot.flatten()
    flat_logits = logits.flatten()
    logits = jax.nn.log_softmax(flat_logits)
    logits_selected= logits*jax.lax.stop_gradient(flat_onehot)
    return jnp.nansum(logits_selected)

def categorical_per_feature(rng, logits:H2MG) -> H2MG:
    # TODO: check addr
    flat_tree, tree_strcut = jax.tree_util.tree_flatten(logits)
    f_idx = []
    for v in flat_tree:
        rng, subkey = jax.random.split(rng, 2)
        f_idx.append(jax.random.categorical(key=subkey, logits=v))
    # flat_idx = map(jax.random.categorical, list(zip(flat_tree, jax.random.split(rng, len(flat_tree))))) # categorical axis=-1==1
    return jax.tree_util.tree_unflatten(tree_strcut, f_idx)

def categorical_per_feature_logprob(x_idx: H2MG, logits: H2MG) -> float:
    logits= logits.apply(jax.nn.log_softmax)# jax.nn.log_softmax(logits, axis=-1)
    # selected_logits = map_to_features(lambda logits, x_i: jnp.take_along_axis(logits, jnp.expand_dims(x_idx,1), axis=-1), logits, x_idx)
    # selected_logits = jnp.take_along_axis(logits, jnp.expand_dims(x_idx,1), axis=-1)
    flat_logits_tree, tree__logits_strcut = jax.tree_util.tree_flatten(logits)
    flat_x_tree, tree_x_strcut = jax.tree_util.tree_flatten(x_idx)
    res = []
    for l, x_i in zip(flat_logits_tree, flat_x_tree):
        tmp = jnp.take_along_axis(l, jnp.expand_dims(x_i,1), axis=-1)
        res.append(tmp)

    selected_logits = jax.tree_util.tree_unflatten(tree__logits_strcut, res)

    return selected_logits.nansum()

