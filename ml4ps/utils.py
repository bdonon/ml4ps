import numpy as np


# def collate_dict_old(data):
#     """Transforms a list of dictionaries into a dictionary whose values are tensors with an additional dimension."""
#     return {k: {f: np.array([d[k][f] for d in data]) for f in data[0][k].keys()} for k in data[0].keys()}


def collate_dict(data):
    """Transforms a list of data x (nested dictionaries) into a nested dictionary of batched tensors."""
    if isinstance(data[0], dict):
        r = {}
        for k in data[0].keys():
            r[k] = collate_dict([sample[k] for sample in data])
    else:
        r = np.array(data)
    return r


def collate_power_grid(data):
    """Collates pairs `(x, nets)`, by only collating `x` and leaving `nets` untouched."""
    x, nets = zip(*data)
    return collate_dict(x), nets


# def collate_power_grid_old(data):
#     """Collates tuples `(a, x, nets)`, by only collating `a` and `x` and leaving `nets` untouched."""
#     a, x, network = zip(*data)
#     return collate_dict(a), collate_dict(x), network


def separate_dict(data):
    """Transforms a dict of batched tensors into a list of dicts that have single tensors as values."""
    r = {}
    for k in data.keys():
        if isinstance(data[k], dict):
            r[k] = separate_dict(data[k])
        else:
            r[k] = data[k]
    n_batch = max([len(list_) for list_ in r.values()])
    return [{key: r[key][i] for key in r.keys()} for i in range(n_batch)]


# def separate_dict_old(data):
#     """Transforms a dict of batched tensors into a list of dicts that have single tensors as values."""
#     elem = list(list(data.values())[0].values())[0]
#     batch_size = np.shape(elem)[0]
#     return [{k: {f: data[k][f][i] for f in v} for k, v in data.items()} for i in range(batch_size)]

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


# def clean_dict_old(v):
#     """Cleans a dictionary of tensors by deleting keys whose values are empty."""
#     keys_to_erase = []
#     for k, v_k in v.items():
#         keys_to_erase_k = []
#         for f, v_k_f in v_k.items():
#             if np.shape(v_k_f)[0] == 0:
#                 keys_to_erase_k.append(f)
#         for f in keys_to_erase_k:
#             del v_k[f]
#         if not v_k:
#             keys_to_erase.append(k)
#     for k in keys_to_erase:
#         del v[k]
#     return v


# def build_unique_id_dict(table_dict, addresses):
#     """Builds a dictionary to convert `str` indices into unique `int`."""
#     all_addresses = [list(table_dict[k][f].values.astype(str)) for k, v in addresses.items() for f in v]
#     unique_addresses = list(np.unique(np.concatenate(all_addresses)))
#     return {address: i for i, address in enumerate(unique_addresses)}


def convert_addresses_to_integers(x, address_names):
    all_addresses = []
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

    #
    #     for key, val in x.items():
    #         if "address" in x[key].keys():
    #             for a in x[key]["address"].keys():
    #                 x[key]["address"][a] = np.vectorize(str_to_int.get)(x[key]["address"][a])
    #
    #
    #     if "address" in x[key].keys():
    #         for address_list in x[key]["address"].values():
    #             all_addresses.append(address_list)
    #         #all_addresses.append([list(address_list) for address_list in x[key]["address"].values()])
    # if all_addresses:
    #     unique_addresses = list(np.unique(np.concatenate(all_addresses)))
    #     str_to_int = {address: i for i, address in enumerate(unique_addresses)}
    #     for key, val in x.items():
    #         if "address" in x[key].keys():
    #             for a in x[key]["address"].keys():
    #                 x[key]["address"][a] = np.vectorize(str_to_int.get)(x[key]["address"][a])
    #     #return x


def assert_substructure(a, b):
    """Asserts that `a` is a substructure of `b`."""
    if isinstance(a, dict):
        for k in a.keys():
            assert (k in b.keys())
            assert_substructure(a[k], b[k])
    elif isinstance(a, list):
        assert ((sorted(a) == sorted(b)) and (len(a) == len(b)))


def get_n_obj(x):
    """Returns a dictionary that counts the amount of objects of each class."""
    r = {}
    for k in x.keys():
        current_max = 0
        if k == 'global':
            for f in x[k].keys():
                current_max = max(current_max, np.shape(x[k][f])[1])
        else:
            # for f in x[k].keys():
            # if f=='address':
            for a in x[k]['address'].keys():
                current_max = max(current_max, np.shape(x[k]['address'][a])[1])
            for f in x[k]['features'].keys():
                current_max = max(current_max, np.shape(x[k]['features'][f])[1])
        r[k] = current_max
    return r


def get_n_obj_old(x):
    """Returns a dictionary that counts the amount of objects of each class."""
    return {k: np.max([np.shape(x_k_f)[1] for f, x_k_f in x_k.items()]) for k, x_k in x.items()}
