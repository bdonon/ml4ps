import numpy as np
import jax.numpy as jnp


def stack_batch(x_list):
    n_obj = np.max([np.shape(x)[-1] for x in x_list])
    return np.stack([np.pad(x.astype(float), [(0, n_obj-np.shape(x)[-1])], mode='constant', constant_values=np.nan) for x in x_list])


def collate_dict(x_batch):
    r = {}
    for k in x_batch[0].keys():
        if isinstance(x_batch[0][k], dict):
            r[k] = {}
            for f in x_batch[0][k].keys():
                r[k][f] = stack_batch([x[k][f] for x in x_batch])
        else:
            r[k] = stack_batch([x[k] for x in x_batch])
    return r

#def collate_dict(x_batch, pad_value=np.nan):
#    """Collates nested dictionaries and pads missing values using `pad_value`."""
#    max_n_obj = get_max_n_obj(x_batch)
#    x_batch = pad_missing_values(x_batch, max_n_obj, pad_value)
#    return {k: {f: np.stack([x[k][f] for x in x_batch]) for f in x_batch[0][k].keys()} for k in x_batch[0].keys()}


def collate_power_grid(data, **kwargs):
    """Collates lists of pairs `(x, nets)`, by only collating `x` and leaving `nets` untouched.

    In the case where samples have different number of objects, additional objects are created and are associated
    with the value specified in `pad_value`.
    """
    x_batch, nets = zip(*data)
    return collate_dict(x_batch, **kwargs), nets


def separate_dict(data):
    """Transforms a dict of batched tensors into a list of dicts that have single tensors as values.

    TODO : clean this
    """


    r = {}
    for k in data.keys():
        if isinstance(data[k], dict):
            r[k] = separate_dict(data[k])
        else:
            r[k] = data[k]
    n_batch = max([len(list_) for list_ in r.values()])

    results = []
    for i in range(n_batch):
        dict_ = {}
        for key in r.keys():
            if isinstance(r[key], np.ndarray):
                if len(r[key][i].shape) > 1:
                    dict_[key] = r[key][i]
                else:
                    dict_[key] = r[key][i][~np.isnan(r[key][i])]
            elif isinstance(r[key], jnp.DeviceArray):
                temp = np.array(r[key][i])
                if len(temp.shape) > 1:
                    dict_[key] = temp
                else:
                    dict_[key] = temp[~np.isnan(temp)]
            else:
                dict_[key] = r[key][i]

        results.append(dict_)
    return results


def clean_dict(data):
    """Cleans a dictionary of tensors by deleting keys whose values are empty."""
    keys_to_erase = []
    for k in data.keys():
        if isinstance(data[k], dict):
            clean_dict(data[k])
            if not data[k]:  # Check is data[k] has been emptied
                keys_to_erase.append(k)
        elif isinstance(data[k], list):
            if len(data[k]) == 0:
                keys_to_erase.append(k)
        else:
            if data[k].size == 0:
                keys_to_erase.append(k)
    for k in keys_to_erase:
        del data[k]
    return data


def convert_addresses_to_integers(x, address_names):
    """Converts `str` addresses into a unique integer id.

    Only addresses specified in `address_names` are considered for defining the mapping from `str` address to
    unique integer id.
    """
    all_addresses, unique_addresses = [], []
    for object_name, object_address_names in address_names.items():
        if object_name in x.keys():
            for object_address_name in object_address_names:
                all_addresses.append(x[object_name][object_address_name])
    if all_addresses:
        unique_addresses = list(np.unique(np.concatenate(all_addresses)))
        str_to_int = {address: i for i, address in enumerate(unique_addresses)}
        converter = np.vectorize(str_to_int.get)
        for object_name, object_address_names in address_names.items():
            if object_name in x.keys():
                for object_address_name in object_address_names:
                    x[object_name][object_address_name] = converter(x[object_name][object_address_name])


    n_unique_addresses = len(unique_addresses)
    x['h_g'] = np.zeros([1, 1])
    x['h_v'] = np.zeros([n_unique_addresses, 1])



def assert_substructure(a, b):
    """Asserts that `a` is a substructure of `b`."""
    if isinstance(a, dict):
        for k in a.keys():
            assert (k in b.keys())
            assert_substructure(a[k], b[k])
    elif isinstance(a, list):
        assert set(a).issubset(set(b))


def get_max_n_obj(x_batch):
    """Returns a dictionary that counts the amount of objects of each class."""
    max_n_obj = {}
    for x in x_batch:
        for k in x.keys():
            for f in x[k].keys():
                current_n_obj = x[k][f].shape[-1]
                max_n_obj[k] = max(max_n_obj.get(k, 0), current_n_obj)
    return max_n_obj


def pad_missing_values(x_batch, max_n_obj, pad_value):
    """Adds `pad_value` to ensure that all samples have the same amount of objects."""
    for x in x_batch:
        for k in x.keys():
            for f in x[k].keys():
                current_shape = x[k][f].shape
                missing_objects = max_n_obj[k] - current_shape[-1]
                if missing_objects > 0:
                    x[k][f] = np.concatenate([x[k][f], pad_value * np.ones([*current_shape[-1:], missing_objects])])
    return x_batch


def get_n_obj(x):
    """Returns a dictionary that counts the amount of objects of each class."""
    r = {}
    for k, x_k in x.items():
        if isinstance(x_k, dict):
            r[k] = np.max([np.shape(x_k_f)[1] for f, x_k_f in x_k.items()])
    return r

    #return {k: np.max([np.shape(x_k_f)[1] for f, x_k_f in x_k.items()]) for k, x_k in x.items()}
