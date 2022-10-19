import tqdm
import pickle
import os


def pickle_dataset(source_dir, target_dir, backend, data_structure=None, save_network=True):
    """Converts a dataset into a pickle version.

    It imports each `net` in `source_dir`, extracts the desired features `x`, and saves the pair `(x, net)` in a
    .pkl file in `target_dir`. The resulting dataset can then be used by a PowerGridDataset, by setting the
    keyword argument `pickle` to True.

    Example :
        >>> import ml4ps as mp
        >>> backend = mp.PandaPowerBackend()
        >>> pickle_dataset('data/train', 'data/train_pkl', backend=backend)
        >>> dataset = mp.PowerGridDataset('data/train_pkl', backend=backend, pickle=True)

    **Note**
    Depending on the backend used, this may provide a consequent speed-up, or slow the data processing pipeline.
    It appears that backends that are pure python are more likely to benefit from this pickle conversion.
    In the case of the PandaPowerBackend, it has been observed to provide a 10x speed up in data loading.

    Args:
        source_dir (:obj:`dict`): Path to the source dataset. There should be a series of power grid instances
            compatible with the provided `backend` inside this directory. Note that train and test directories
            should be converted separately.
        target_dir (:obj:`dict`): Path to the target dataset to be created. The .pkl files are stored in this
            path, and use the exact same file names as in the `source_dir`. We recommend to reuse the same name
            as the `source_dir` plus a suffix "_pkl".
        backend (:obj:`ml4ps.backend.Backend`): ML4PS backend instance. It provides a mean of reading the data,
            checking the validity of the data_structure and extracting the feature values.
        data_structure (:obj:`dict` of `dict` of `list` of `str`, optional): Details the feature names and address
            names that should be extracted from `net`. By default, it uses the valid_data_structure of the
            provided backend.
        save_network (:obj:`bool`, optional): If true, saves both the data `x` and the power grid instance `net`
            in the .pkl file. Otherwise, it only saves the data `x` in the .pkl file. In that case, when creating
            a PowerGridDataSet, the keyword argument `return_network` should be set to False.
    """
    if data_structure is None:
        data_structure = backend.valid_data_structure
    backend.check_data_structure(data_structure)
    file_path_list = backend.get_valid_files(source_dir)
    os.mkdir(target_dir)

    for file_path in tqdm.tqdm(file_path_list, desc='Converting dataset to pickle. '):
        file_name = os.path.basename(file_path)
        case_name = os.path.splitext(file_name)[0]
        net = backend.load_network(file_path)
        x = backend.get_data_network(net, data_structure)
        if save_network:
            data = {'x': x, 'net': net}
        else:
            data = {'x': x}

        target_file_path = os.path.join(target_dir, case_name+'.pkl')
        with open(target_file_path, 'wb') as fp:
            pickle.dump(data, fp)
