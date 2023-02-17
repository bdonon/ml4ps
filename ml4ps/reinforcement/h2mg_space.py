from gymnasium import spaces
import gymnasium as gym
from ml4ps.h2mg import map_to_features, H2MG, empty_h2mg
from copy import deepcopy
from collections import OrderedDict

class H2MGSpace(spaces.Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def shape(self):
        return map_to_features(lambda x: x.shape, [self])
    
    @property
    def high(self):
        return map_to_features(lambda x: x.high, [self])
    
    @property
    def low(self):
        return map_to_features(lambda x: x.low, [self])
    
    @property
    def size(self):
        return map_to_features(lambda x: x.size, [self])
    
    def _feature_dimension(self, feature):
        if len(feature.shape) == 1:
            return 1
        elif len(feature.shape) == 2:
            return feature.shape[-1]
        else:
            raise ValueError(f"Invalid shape: {feature.shape}")

    @property
    def feature_dimension(self):
        return map_to_features(self._feature_dimension, [self])
    
    @property
    def continuous(self) -> 'H2MGSpace':
        mask = map_to_features(lambda x: isinstance(x, spaces.Box), [self])
        return _fill_on_mask(self, mask)
    
    @property
    def multi_discrete(self) -> 'H2MGSpace':
        mask = map_to_features(lambda x: isinstance(x, spaces.MultiDiscrete), [self])
        return _fill_on_mask(self, mask)

    @property
    def multi_binary(self) -> 'H2MGSpace':
        mask = map_to_features(lambda x: isinstance(x, spaces.MultiBinary), [self])
        return _fill_on_mask(self, mask)

    
    def combine(self, other: 'H2MGSpace') -> 'H2MGSpace':
        mask = map_to_features(lambda _: True, [self])
        other_mask = map_to_features(lambda _: True, [other])
        x = _fill_on_mask(self, mask)
        x = _fill_on_mask(other, other_mask, x)
        return x

    @property
    def h2mg_struct(self):
        return self.feature_dimension



@gym.vector.utils.spaces.batch_space.register(H2MGSpace)
def _my_batch(space, n=1):
        return H2MGSpace(
            OrderedDict(
                [
                    (key, gym.vector.utils.spaces.batch_space(subspace, n=n))
                    for (key, subspace) in space.spaces.items()
                ]
            ),
            seed=deepcopy(space.np_random),
        )
@gym.vector.utils.shared_memory.create_shared_memory.register(H2MGSpace)
def _create_dict_shared_memory(space, n=1, ctx=None):
    return H2MG(
        [
            (key, gym.vector.utils.shared_memory.create_shared_memory(subspace, n=n, ctx=ctx))
            for (key, subspace) in space.spaces.items()
        ]
    )

@gym.vector.utils.shared_memory.read_from_shared_memory.register(H2MGSpace)
def _my_read(space, shared_memory, n: int = 1):
        return H2MG(
        [
            (key, gym.vector.utils.shared_memory.read_from_shared_memory(subspace, shared_memory[key], n=n))
            for (key, subspace) in space.spaces.items()
        ]
    )


def _fill_on_mask(fill_h2mg: H2MG, mask: H2MG, output_h2mg=None):
    if output_h2mg is None:
        output_h2mg = dict()
    for local_key, obj_name, feat_name, value in mask.local_features_iterator:
        if value:
            if local_key not in output_h2mg:
                output_h2mg[local_key] = spaces.Dict()
            if obj_name not in output_h2mg[local_key]:
                output_h2mg[local_key][obj_name] = spaces.Dict()
            output_h2mg[local_key][obj_name][feat_name] = fill_h2mg[local_key][obj_name][feat_name]

    for global_key,  feat_name, value in mask.global_features_iterator:
        if value:
            if global_key not in output_h2mg:
                output_h2mg[global_key] = spaces.Dict()
            output_h2mg[global_key][feat_name] = fill_h2mg[global_key][feat_name]

    for local_key, obj_name, addr_name, value in mask.local_addresses_iterator:
        if value:
            if local_key not in output_h2mg:
                output_h2mg[local_key] = spaces.Dict()
            if obj_name not in output_h2mg[local_key]:
                output_h2mg[local_key][obj_name] = spaces.Dict()
            output_h2mg[local_key][obj_name][addr_name] = fill_h2mg[local_key][obj_name][addr_name]

    for all_addr_key, value in mask.all_addresses_iterator:
        if value:
            output_h2mg[all_addr_key] = fill_h2mg[all_addr_key]
    return H2MGSpace(output_h2mg)