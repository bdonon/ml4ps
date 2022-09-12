import numpy as np
import math

class Iterator:
    """Iterates through a dataset of power grids.

        An iterator allows to iteratively retrieve batches of power networks,
        as well as their addresses and features. It returns a tuple
        (a_concat, x_concat, net_batch), where:

        -   net_batch is a list of list of power network instances related to
            the iterator backend. Returning this part is useful if the training
            loop also requires modifies the network batch to perform some
            electrotechnical computation.
        -   x_concat is a nested dictionary that has a structure defined by
            the features attribute of the iterator. For a given pair of
            electrotechnical object class (e.g. 'gen') and attribute (e.g.
            'p_min') specified in the features attribute of the iterator, it
            gives a tensor of shape [n_batch, n_obj, n_window], where n_batch
            is the batch size, n_obj is the amount of such object in the
            power network instance (e.g. amount of 'gen'), and n_window is
            the time window size. This tensor contains the corresponding
            values extracted from net_batch
        -   a_concat is a nested dictionary that has a structure defined by
            the features attribute of the iterator. For a given pair of
            electrotechnical object class (e.g. 'gen') and address (e.g.
            'bus_id') specified in the addresses attribute of the iterator, it
            gives a tensor of shape [n_batch, n_obj, 1], where n_batch
            is the batch size, n_obj is the amount of such object in the
            power network instance (e.g. amount of 'gen'), and n_window is
            the time window size. This tensor an integer representation of
            the addresses.

        Attributes:
            files: A list of string specifying valid file names.
            backend: A backend implementation that inherits from AbstractBackend.
            features: A dict of list of strings specifying the features that
                should be retrieved in the output x of the iterator.
            addresses: A dict of list of strings specifying the features that
                should be retrieved in the output x of the iterator.
            series_length: An integer that defines the coherence length of
                time series.
            time_window: An integer that defines the length of time windows.
            batch_size: An integer that defines the size of each batch that should
                be output by the iterator.
            shuffle: A boolean that specifies if the dataset should be shuffled.
            keys: A list containing all the keys used in either features or
                addresses.
            window_files: A list of list of file names of exactly time_window
                size.
            batch_files: A list of list of list of file names, that break
                window_files into batches of batch_size size.
            length: An integer defining the length of the list of batches.
    """

    def __init__(self, file_list, backend, **kwargs):
        """Inits interface."""

        self.file_list = file_list
        self.backend = backend

        self.features = kwargs.get("features", self.backend.valid_features)
        self.addresses = kwargs.get("addresses", self.backend.valid_addresses)
        self.series_length = kwargs.get("series_length", 1)
        self.time_window = kwargs.get("time_window", 1)
        self.batch_size = kwargs.get("batch_size", 10)
        self.shuffle = kwargs.get("shuffle", True)

        self.backend.check_features(self.features)
        self.backend.check_addresses(self.addresses)
        self.keys = list(set(list(self.features.keys()) + list(self.addresses.keys())))

        self.window_files = None
        self.batch_files = None
        self.length = None
        self.initialize_batch_order()

        if self.time_window > self.series_length:
            raise ValueError('Time window should be larger that series length.')

    def initialize_batch_order(self):
        """Initializes the batch list of time windows."""
        #self.window_files = self.split_window(self.file_list)
        if self.shuffle:
            #np.random.shuffle(self.window_files)
            np.random.shuffle(self.file_list)
        #self.batch_files = self.split_batch(self.window_files)
        self.batch_files = self.split_batch(self.file_list)
        self.length = len(self.batch_files)

    def split_window(self, file_list):
        """Splits a list of file names into a list of sublists.

        Splits a list of file names into overlapping lists of exactly
        self.time_window size, and respect the time coherence of the
        various time series.
        For instance, consider the following input:

        file_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

        Assume that the iterator has a series_length of 3 and a
        time_window of 2.
        Since series_length is set to 3, it considers ['A', 'B', 'C']
        and ['D', 'E', 'F'] as coherent time series, and discards ['G'],
        as its length is smaller than series_length.
        Then, it scans through each of these time series and takes all
        rolling windows of size time_window. It thus returns the
        following:

        [['A', 'B'], ['B', 'C'], ['D', 'E'], ['E', 'F']]

        Args:
          file_list: A list of strings defining paths to valid files.

        Returns:
          A list of lists of strings defining paths to valid files
        """
        n_files = math.floor(len(file_list) / self.series_length) * self.series_length
        sl = self.series_length
        tw = self.time_window
        window_files = [file_list[i + j:i + j + tw] for i in range(0, n_files, sl) for j in range(0, sl - tw + 1)]
        return window_files

    def split_batch(self, window_files):
        """Splits the list window_files into chunks of size batch_size."""
        bs = self.batch_size
        nb = len(window_files)
        return [window_files[i:i + bs] for i in range(0, nb, bs)]

    def __iter__(self):
        """Initialization of the iterations."""
        self.current_batch_id = 0
        self.initialize_batch_order()
        return self

    def __len__(self):
        """Length of the iterator, for tqdm compatibility."""
        return self.length

    def __next__(self):
        """Gets current batch, and increments current_batch_id."""
        if self.current_batch_id < len(self.batch_files):
            current_batch_files = self.batch_files[self.current_batch_id]
            print(current_batch_files)
            a_concat, x_concat, net_batch = self.backend.get_batch(current_batch_files, addresses=self.addresses,
                                                                   features=self.features)
            self.current_batch_id += 1
        else:
            raise StopIteration
        return a_concat, x_concat, net_batch

