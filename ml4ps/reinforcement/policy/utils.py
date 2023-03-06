from collections import defaultdict
from typing import Any, Callable, Dict, Tuple

import gymnasium
from gymnasium import spaces
from ml4ps import h2mg
from jax import numpy as jnp

def add_prefix(x, prefix):
    return transform_feature_names(x, lambda feat_name: prefix+feat_name)


def remove_prefix(x, prefix):
    return transform_feature_names(x, lambda feat_name: feat_name.removeprefix(prefix))


def tr_feat(feat_names, fn):
    if isinstance(feat_names, list):
        return list(map(fn, feat_names))
    elif isinstance(feat_names, dict):
        return {fn(feat): value for feat, value in feat_names.items()}


def transform_feature_names(_x, fn: Callable):
    x = _x.copy()
    if "local_features" in x:
        x |= {"local_features": {obj_name: tr_feat(
            obj, fn) for obj_name, obj in x["local_features"].items()}}
    if "global_features" in x:
        x |= {"global_features": tr_feat(x["global_features"], fn)}
    return x


def slice_with_prefix(_x, prefix):
    x = _x.copy()
    if "local_features" in x:
        x |= {"local_features": {obj_name: {feat.removeprefix(prefix): value for feat, value in obj.items(
        ) if feat.startswith(prefix)} for obj_name, obj in x["local_features"].items()}}
    if "global_features" in x:
        x |= {"global_features": {feat.removeprefix(
            prefix): value for feat, value in x["global_features"].items() if feat.startswith(prefix)}}
    return x


def combine_space(a, b):
    x = h2mg.empty_h2mg()
    for local_key, obj_name, feat_name, value in h2mg.local_features_iterator(a):
        x[local_key][obj_name][feat_name] = value
    for local_key, obj_name, feat_name, value in h2mg.local_features_iterator(b):
        x[local_key][obj_name][feat_name] = value

    for global_key,  feat_name, value in h2mg.global_features_iterator(a):
        x[global_key][feat_name] = value
    for global_key,  feat_name, value in h2mg.global_features_iterator(b):
        x[global_key][feat_name] = value

    for local_key, obj_name, addr_name, value in h2mg.local_addresses_iterator(a):
        x[local_key][obj_name][addr_name] = value

    for all_addr_key, value in h2mg.all_addresses_iterator(a):
        x[all_addr_key][value] = value
    return x


def combine_feature_names(feat_a, feat_b):
    new_feat_a = defaultdict(lambda: defaultdict(list))
    for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(feat_a):
        new_feat_a[local_key][obj_name].append(feat_name)
    for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(feat_b):
        new_feat_a[local_key][obj_name].append(feat_name)

    for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(feat_b):
        new_feat_a[local_key][obj_name] = list(
            set(new_feat_a[local_key][obj_name]))

    new_feat_a[h2mg.H2MGCategories.GLOBAL_FEATURES.value] = list()
    for global_key, feat_name in h2mg.global_feature_names_iterator(feat_a):
        new_feat_a[global_key].append(feat_name)
    for global_key, feat_name in h2mg.global_feature_names_iterator(feat_b):
        new_feat_a[global_key].append(feat_name)
    new_feat_a[h2mg.H2MGCategories.GLOBAL_FEATURES.value] = list(
        set(new_feat_a[h2mg.H2MGCategories.GLOBAL_FEATURES.value]))
    return new_feat_a


def space_to_feature_names(space: spaces.Space):
    feat_names = {}
    if "local_addresses" in list(space.keys()):
        feat_names |= {"local_addresses": {
            k: list(v) for k, v in space["local_addresses"].items()}}
    if "local_features" in list(space.keys()):
        feat_names |= {"local_features": {
            k: list(v) for k, v in space["local_features"].items()}}
    if "global_features" in list(space.keys()):
        feat_names |= {"global_features": list(space["global_features"].keys())}
    return feat_names


def flatten_dict(d, flat_dim=None):
    if flat_dim is None:
        flat_dim = sum(v.size for v in h2mg.features_iterator(d))
    flat_action = jnp.zeros(shape=flat_dim)
    i=0
    for v in h2mg.features_iterator(d):
        flat_action = flat_action.at[i:i+len(v)].set(v)
        i+=len(v)
    return flat_action

def unflatten_like(x, d):
    res = h2mg.empty_like(d)
    i = 0
    for key, obj_name, feat_name, value in h2mg.local_features_iterator(d):
        res[key][obj_name][feat_name] = x[i:i+value.size]
        i+=value.size
    for key, feat_name, value in h2mg.global_features_iterator(d):
        res[key][feat_name] = x[i:i+value.size]
        i+=value.size
    if i != x.size:
        print(i)
        print(x.size)
        raise ValueError
    return res

def get_single_action_space(env: gymnasium.Env) -> gymnasium.Space:
    if isinstance(env, gymnasium.vector.VectorEnv):
        return env.single_action_space
    else:
        return env.action_space