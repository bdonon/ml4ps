from collections import defaultdict
from enum import Enum
from typing import Callable, Iterator, List, Dict

import jax.numpy as jnp

from gymnasium import spaces

class H2MGSpace(spaces.Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class H2MG(dict):

    def __init__(self, data):
        super().__init__(data)
    
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
    if local_key in h2mg:
        for obj_name in h2mg[local_key]:
            for feat_name, value in h2mg[local_key][obj_name].items():
                yield local_key, obj_name, feat_name, value


def global_features_iterator(h2mg) -> Iterator:
    global_key = H2MGCategories.GLOBAL_FEATURES.value
    if H2MGCategories.GLOBAL_FEATURES.value in h2mg:
        for feat_name, value in h2mg[global_key].items():
            yield global_key,  feat_name, value


def local_addresses_iterator(h2mg) -> Iterator:
    local_key = H2MGCategories.LOCAL_ADDRESSES.value
    if local_key in h2mg:
        for obj_name in h2mg[local_key]:
            for addr_name, value in h2mg[local_key][obj_name].items():
                yield local_key, obj_name, addr_name, value


def all_addresses_iterator(h2mg) -> Iterator:
    all_addr_key = H2MGCategories.ALL_ADDRESSES.value
    if all_addr_key in h2mg:
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
            new_h2mg[key][obj_name][feat_name] = jnp.empty_like(jnp.array(value))

    for key, feat_name, value in global_features_iterator(h2mg):
        if key not in new_h2mg:
            new_h2mg[key] = {}
        if feat_name not in new_h2mg[key]:
            new_h2mg[key][feat_name] = jnp.empty_like(jnp.array(value))

    for key, obj_name, addr_name, value in local_addresses_iterator(h2mg):
        if key not in new_h2mg:
            new_h2mg[key] = {}
        if obj_name not in new_h2mg[key]:
            new_h2mg[key][obj_name] = {}
        if addr_name not in new_h2mg[key][obj_name]:
            new_h2mg[key][obj_name][addr_name] = value

    for key, value in all_addresses_iterator(h2mg):
        new_h2mg[key] = value

    return new_h2mg

def collate_h2mgs_features(h2mgs_list):
    def collate_arrays(*args):
        return jnp.array(list(args))
    return map_to_features(collate_arrays, h2mgs_list)

def h2mg_map(fn: Callable, args_h2mg: List=None, local_features: bool=True, global_features: bool=True, local_addresses: bool=False, all_addresses: bool=False, check_compat: bool=False) -> Dict:
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

    return results

def map_to_features(fn, args_h2mg, check_compat=False):
    return h2mg_map(fn=fn, args_h2mg=args_h2mg, check_compat=check_compat)

def map_to_all(fn: Callable, args_h2mg: List, check_compat=False):
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