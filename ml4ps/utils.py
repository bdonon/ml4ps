import numpy as np


def collate_dict(data):
    """Transforms a list of dictionaries into a dictionary whose values are tensors with an additional dimension."""
    return {k: {f: np.array([d[k][f] for d in data]) for f in data[0][k].keys()} for k in data[0].keys()}


def collate_power_grid(data):
    """Collates tuples `(a, x, nets)`, by only collating `a` and `x` and leaving `nets` untouched."""
    a, x, network = zip(*data)
    return collate_dict(a), collate_dict(x), network


def separate_dict(data):
    """Transforms a dict of batched tensors into a list of dicts that have single tensors as values."""
    elem = list(list(data.values())[0].values())[0]
    batch_size = np.shape(elem)[0]
    return [{k: {f: data[k][f][i] for f in v} for k, v in data.items()} for i in range(batch_size)]


def clean_dict(v):
    """Cleans a dictionary of tensors by deleting keys whose values are empty."""
    keys_to_erase = []
    for k, v_k in v.items():
        keys_to_erase_k = []
        for f, v_k_f in v_k.items():
            if np.shape(v_k_f)[0] == 0:
                keys_to_erase_k.append(f)
        for f in keys_to_erase_k:
            del v_k[f]
        if not v_k:
            keys_to_erase.append(k)
    for k in keys_to_erase:
        del v[k]
    return v


def build_unique_id_dict(table_dict, addresses):
    """Builds a dictionary to convert `str` indices into unique `int`."""
    all_addresses = [list(table_dict[k][f].values.astype(str)) for k, v in addresses.items() for f in v]
    unique_addresses = list(np.unique(np.concatenate(all_addresses)))
    return {address: i for i, address in enumerate(unique_addresses)}