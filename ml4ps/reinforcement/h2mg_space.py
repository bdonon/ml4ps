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
    def continuous(self):
        mask = map_to_features(lambda x: isinstance(x, spaces.Box), [self])
        x = dict()
        for local_key, obj_name, feat_name, value in mask.local_features_iterator:
            if value:
                if local_key not in x:
                    x[local_key] = spaces.Dict()
                if obj_name not in x[local_key]:
                    x[local_key][obj_name] = spaces.Dict()
                x[local_key][obj_name][feat_name] = self[local_key][obj_name][feat_name]

        for global_key,  feat_name, value in mask.global_features_iterator:
            if value:
                if global_key not in x:
                    x[global_key] = spaces.Dict()
                x[global_key][feat_name] = self[global_key][feat_name]

        for local_key, obj_name, addr_name, value in mask.local_addresses_iterator:
            if value:
                if local_key not in x:
                    x[local_key] = spaces.Dict()
                if obj_name not in x[local_key]:
                    x[local_key][obj_name] = spaces.Dict()
                x[local_key][obj_name][addr_name] = self[local_key][obj_name][addr_name]

        for all_addr_key, value in mask.all_addresses_iterator:
            if value:
                x[all_addr_key] = self[all_addr_key]
        return H2MGSpace(x)
    
    @property
    def multi_discrete(self):
        mask = map_to_features(lambda x: isinstance(x, spaces.MultiDiscrete), [self])
        x = dict()
        for local_key, obj_name, feat_name, value in mask.local_features_iterator:
            if value:
                if local_key not in x:
                    x[local_key] = spaces.Dict()
                if obj_name not in x[local_key]:
                    x[local_key][obj_name] = spaces.Dict()
                x[local_key][obj_name][feat_name] = self[local_key][obj_name][feat_name]

        for global_key,  feat_name, value in mask.global_features_iterator:
            if value:
                if global_key not in x:
                    x[global_key] = spaces.Dict()
                x[global_key][feat_name] = self[global_key][feat_name]

        for local_key, obj_name, addr_name, value in mask.local_addresses_iterator:
            if value:
                if local_key not in x:
                    x[local_key] = spaces.Dict()
                if obj_name not in x[local_key]:
                    x[local_key][obj_name] = spaces.Dict()
                x[local_key][obj_name][addr_name] = self[local_key][obj_name][addr_name]

        for all_addr_key, value in mask.all_addresses_iterator:
            if value:
                x[all_addr_key] = self[all_addr_key]
        return H2MGSpace(x)

    @property
    def multi_binary(self):
        mask = map_to_features(lambda x: isinstance(x, spaces.MultiBinary), [self])
        x = dict()
        for local_key, obj_name, feat_name, value in mask.local_features_iterator:
            if value:
                if local_key not in x:
                    x[local_key] = spaces.Dict()
                if obj_name not in x[local_key]:
                    x[local_key][obj_name] = spaces.Dict()
                x[local_key][obj_name][feat_name] = self[local_key][obj_name][feat_name]

        for global_key,  feat_name, value in mask.global_features_iterator:
            if value:
                if global_key not in x:
                    x[global_key] = spaces.Dict()
                x[global_key][feat_name] = self[global_key][feat_name]

        for local_key, obj_name, addr_name, value in mask.local_addresses_iterator:
            if value:
                if local_key not in x:
                    x[local_key] = spaces.Dict()
                if obj_name not in x[local_key]:
                    x[local_key][obj_name] = spaces.Dict()
                x[local_key][obj_name][addr_name] = self[local_key][obj_name][addr_name]

        for all_addr_key, value in mask.all_addresses_iterator:
            if value:
                x[all_addr_key] = self[all_addr_key]
        return H2MGSpace(x)

    @property
    def h2mg_struct(self):
        return self.feature_dimension



@gym.vector.utils.spaces.batch_space.register(H2MGSpace)
def _my_batch(space, n=1):
        print('batching')
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
    print("Creating shared memory")
    return H2MG(
        [
            (key, gym.vector.utils.shared_memory.create_shared_memory(subspace, n=n, ctx=ctx))
            for (key, subspace) in space.spaces.items()
        ]
    )

@gym.vector.utils.shared_memory.read_from_shared_memory.register(H2MGSpace)
def _my_read(space, shared_memory, n: int = 1):
        print("read_from_shared_memory h2mg")
        return H2MG(
        [
            (key, gym.vector.utils.shared_memory.read_from_shared_memory(subspace, shared_memory[key], n=n))
            for (key, subspace) in space.spaces.items()
        ]
    )