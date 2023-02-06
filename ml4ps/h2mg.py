from collections import defaultdict
from enum import Enum
from typing import Callable, Iterator

import numpy as np


class H2MGCategories(Enum):

    LOCAL_FEATURES = "local_features"

    GLOBAL_FEATURES = "global_features"

    LOCAL_ADDRESSES = "local_addresses"

    ALL_ADDRESSES = "all_addresses"


def h2mg(local_features, global_features, local_addresses, all_addresses):
    return {"local_features": local_features,
            "global_features": global_features,
            "local_addresses": local_addresses,
            "all_addresses": all_addresses}


def local_features(h2mg):
    return h2mg["local_features"]


def global_features(h2mg):
    return h2mg["global_features"]


def local_addresses(h2mg):
    return h2mg["local_addresses"]


def all_addresses(h2mg):
    return h2mg["all_addresses"]


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
    for _, _, _, value in local_addresses_iterator(h2mg):
        yield value
    for _,  _, value in global_features_iterator(h2mg):
        yield value


def apply_on_features(fn: Callable, h2mg):
    for key, obj_name, feat_name, value in local_features_iterator(h2mg):
        h2mg[key][obj_name][feat_name] = fn(value)

    for key, feat_name, value in global_features_iterator(h2mg):
        h2mg[key][feat_name] = fn(value)


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
    if set(h2mg[local_key]) != set(h2mg_other[local_key]):
        return False
    for obj_name in h2mg[local_key]:
        if set(h2mg[local_key][obj_name]) != set(h2mg_other[local_key][obj_name]):
            return False

    local_addr_key = H2MGCategories.LOCAL_ADDRESSES.value
    if set(h2mg[local_addr_key]) != set(h2mg_other[local_addr_key]):
        return False
    for feat_name in h2mg[local_addr_key]:
        if set(h2mg[local_addr_key][feat_name]) != set(h2mg_other[local_addr_key][feat_name]):
            return False

    global_key = H2MGCategories.GLOBAL_FEATURES.value
    if set(h2mg[global_key]) != set(h2mg_other[global_key]):
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
            new_h2mg[key][obj_name][feat_name] = np.empty_like(value)

    for key, feat_name, value in global_features_iterator(h2mg):
        if key not in new_h2mg:
            new_h2mg[key] = {}
        if feat_name not in new_h2mg[key]:
            new_h2mg[key][feat_name] = np.empty_like(value)

    for key, obj_name, addr_name, value in local_addresses_iterator(h2mg):
        if key not in new_h2mg:
            new_h2mg[key] = {}
        if obj_name not in new_h2mg[key]:
            new_h2mg[key][obj_name] = {}
        if addr_name not in new_h2mg[key][obj_name]:
            new_h2mg[key][obj_name][addr_name] = value

    new_h2mg[H2MGCategories.ALL_ADDRESSES.value] = all_addresses(h2mg)

    return new_h2mg


def map_to_features(fn, *h2mgs, check_compat=False):
    if check_compat and not all_compatible(*h2mgs):
        raise ValueError
    results = empty_like(h2mgs[0])
    for key, obj_name, feat_name, value in local_features_iterator(results):
        results[key][obj_name][feat_name] = fn(
            *list(map(h2mg_slicer(key, obj_name, feat_name), list(h2mgs))))

    for key, feat_name, value in global_features_iterator(results):
        results[key][feat_name] = fn(
            *list(map(h2mg_slicer(key, None, feat_name), list(h2mgs))))

    return results

def collate_h2mgs_features(h2mgs_list):
    def collate_arrays(*args):
        return np.array(list(args))
    return map_to_features(collate_arrays, *h2mgs_list)

def map_to_all(fn, *h2mgs, check_compat=False):
    if check_compat and not all_compatible(*h2mgs):
        raise ValueError
    results = empty_like(h2mgs[0])
    for key, obj_name, feat_name, value in local_features_iterator(results):
        results[key][obj_name][feat_name] = fn(
            *list(map(h2mg_slicer(key, obj_name, feat_name), list(h2mgs))))

    for key, feat_name, value in global_features_iterator(results):
        results[key][feat_name] = fn(
            *list(map(h2mg_slicer(key, None, feat_name), list(h2mgs))))
    
    for key, obj_name, feat_name, value in local_addresses_iterator(results):
        results[key][obj_name][feat_name] = fn(
            *list(map(h2mg_slicer(key, obj_name, feat_name), list(h2mgs))))
    
    for key, value in all_addresses_iterator(results):
        results[key] = fn(
            *list(map(h2mg_slicer(key, None, None), list(h2mgs))))

    return results

def collate_h2mgs(h2mgs_list):
    def collate_arrays(*args):
        return np.array(list(args))
    return map_to_all(collate_arrays, *h2mgs_list)

def apply_normalization(norm_fns_h2mg, target_h2mg):
    return map_to_features(lambda norm_fn, feature: norm_fn(feature), norm_fns_h2mg, target_h2mg)