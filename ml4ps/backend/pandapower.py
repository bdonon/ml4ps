import pandapower.converter as pc
import pandapower as pp
import pandas as pd
import numpy as np
import mat73
import json
import copy
import os
import re

from pandapower.converter.matpower.from_mpc import _adjust_ppc_indices, _change_ppc_TAP_value
from matpowercaseframes.utils import int_else_float_except_string
from pandapower.converter.matpower.to_mpc import _ppc2mpc
from scipy.io import savemat

from ml4ps.utils import clean_dict, convert_addresses_to_integers
from ml4ps.backend.interface import AbstractBackend


def from_mpc(file_path, format='.mat'):
    """Implementation of from_mpc that supports both .m and .mat files, while allowing matlab 7.3 .m format."""

    def from_mpc73(mpc_file, f_hz=50, validate_conversion=False, **kwargs):
        """Implementation of from_mpc that supports matlab 7.3 .m format."""
        mpc = mat73.loadmat(mpc_file)
        ppc = {"version": "X.X", "baseMVA": mpc["mpc"]["baseMVA"], "bus": mpc["mpc"]["baseMVA"],
            "gen": mpc["mpc"]["gen"], "branch": mpc["mpc"]["branch"], "gencost": mpc["mpc"]["gencost"]}
        _adjust_ppc_indices(ppc)
        _change_ppc_TAP_value(ppc)
        return pc.from_ppc(ppc, f_hz=f_hz, validate_conversion=validate_conversion, **kwargs)

    def from_m(file_path):
        power_grid = pc.from_mpc(file_path)
        pattern = r'mpc\.{}\s*=\s*\[[\n]?(?P<data>.*?)[\n]?\];'.format("shunt_steps")
        with open(file_path) as f:
            match = re.search(pattern, f.read(), re.DOTALL)
        match = match.groupdict().get('data', None)
        match = match.strip("'").strip('"')
        _list = list()
        for line in match.splitlines():
            line = line.split('%')[0]
            line = line.replace(';', '')
            if line.strip():
                _list.append([int_else_float_except_string(s) for s in line.strip().split()])
        power_grid.shunt = pd.DataFrame(_list,
                                        columns=["bus", "q_mvar", "p_mw", "vn_kv", "step", "max_step", "in_service"])
        power_grid.shunt["name"] = ""
        return power_grid

    def load_object_names(power_grid, file_path):
        try:
            name_path = os.path.splitext(file_path)[0] + '.name'
            with open(name_path, 'r') as f:
                name_dict = json.load(f)
            for key, name_list in name_dict.items():
                power_grid.get(key).name = name_list
        except:
            pass

    if format=='.mat':
        try:
            power_grid = pc.from_mpc(file_path)
        except NotImplementedError:
            power_grid = from_mpc73(file_path)
    elif format == '.m':
        power_grid = from_m(file_path)
    load_object_names(power_grid, file_path)
    return power_grid


def to_mpc(net, filepath, format='.mat', **kwargs):
    """Modification of the `to_mpc` implementation of pandapower
    (https://github.com/e2nIEE/pandapower/blob/develop/pandapower/converter/matpower/to_mpc.py)

    The present implementation saves all objects and sets the status of out-of-service
    objects to 0.
    The default implementation deletes objects that are out-of-service, which
    completely alters the object ordering. For visualization purpose, panoramix relies
    heavily on this ordering.
    """

    net = copy.deepcopy(net)

    # Save actual object status
    gen_status = net.gen.in_service.astype(float).values
    ext_grid_status = net.ext_grid.in_service.astype(float).values
    line_status = net.line.in_service.astype(float).values
    trafo_status = net.trafo.in_service.astype(float).values
    ppc_gen_status = np.concatenate([ext_grid_status, gen_status])
    ppc_branch_status = np.concatenate([line_status, trafo_status])

    # Set all objects to be in_service and convert to pypower object
    net.gen.in_service = True
    net.ext_grid.in_service = True
    net.line.in_service = True
    net.trafo.in_service = True
    ppc = pp.converter.to_ppc(net, **kwargs)

    # Manually change the Gen and Branch status to reflect the actual in_service values
    ppc['gen'][:, 7] = ppc_gen_status
    ppc['branch'][:, 10] = ppc_branch_status

    # Get the current step and max step for shunts
    shunt = net.shunt[["bus", "q_mvar", "p_mw", "vn_kv", "step", "max_step", "in_service"]].astype(float).values
    ppc['bus'][:, 4] = 0.
    ppc['bus'][:, 5] = 0.

    # Untouched part
    mpc = dict()
    mpc["mpc"] = _ppc2mpc(ppc)
    if format == '.mat':
        savemat(filepath, mpc)
    elif format == '.m':

        def write_table(f, arr, max_col=None):
            for r in arr:
                for v in r[:max_col]:
                    if v.is_integer():
                        f.write("\t{}".format(v.astype(int)))
                    else:
                        f.write("\t{:.6f}".format(v))
                f.write(";\n")
            f.write("];\n")

        with open(filepath, "w") as f:
            f.write("function mpc = powergrid\n")
            f.write("mpc.version = '2';\n")
            f.write("mpc.baseMVA = 100;\n")
            f.write("mpc.bus = [\n")
            write_table(f, mpc["mpc"]["bus"], max_col=13)
            f.write("mpc.gen = [\n")
            write_table(f, mpc["mpc"]["gen"], max_col=21)
            f.write("mpc.branch = [\n")
            write_table(f, mpc["mpc"]["branch"], max_col=13)
            f.write("%\tbus_i\tQ\tP\tvn_kv\tstep\tmax_step\tstatus\n")
            f.write("mpc.shunt_steps = [\n")
            write_table(f, shunt)
            f.close()


    # Save names
    names = {
        'bus': list(net.bus.name.astype(str).values),
        'gen': list(net.gen.name.astype(str).values),
        'load': list(net.load.name.astype(str).values),
        'line': list(net.line.name.astype(str).values),
        'trafo': list(net.trafo.name.astype(str).values),
        'ext_grid': list(net.ext_grid.name.astype(str).values),
        'shunt': list(net.shunt.name.astype(str).values),
    }
    with open(os.path.splitext(filepath)[0] + '.name', "w") as outfile:
        json.dump(names, outfile)

    return mpc


class PandaPowerBackend(AbstractBackend):
    """Backend implementation that uses `PandaPower <http://www.pandapower.org>`_."""

    valid_extensions = (".json", ".pkl", ".mat")
    valid_address_names = {
        "bus": ["id"], #, "name"],
        "load": ["bus_id"],#["id", "name", "bus_id"],
        "sgen": ["bus_id"],#["id", "name", "bus_id"],
        "gen": ["bus_id"],#["id", "name", "bus_id"],
        "shunt": ["bus_id"],#["id", "name", "bus_id"],
        "ext_grid": ["bus_id"],#["id", "name", "bus_id"],
        "line": ["from_bus_id", "to_bus_id"],#["id", "name", "from_bus_id", "to_bus_id"],
        "trafo": ["hv_bus_id", "lv_bus_id"],#["id", "name", "hv_bus_id", "lv_bus_id"],
    }
    valid_feature_names = {
        "global": ["converged", "f_hz", "sn_mva"],
        "bus": ["in_service", "max_vm_pu", 'min_vm_pu', "vn_kv", "res_vm_pu", "res_va_degree", "res_p_mw",
            "res_q_mvar"],
        "load": ["const_i_percent", "const_z_percent", "controllable", "in_service", "p_mw", "q_mvar", "scaling",
            "sn_mva", "res_p_mw", "res_q_mvar"],
        "sgen": ["controllable", "in_service", "p_mw", "q_mvar", "scaling", "sn_mva", "current_source", "res_p_mw",
            "res_q_mvar"],
        "gen": ["controllable", "in_service", "p_mw", "scaling", "sn_mva", "vm_pu", "slack", "max_p_mw", "min_p_mw",
            "max_q_mvar", "min_q_mvar", "slack_weight", "res_p_mw", "res_q_mvar", "res_va_degree", "res_vm_pu"],
        "shunt": ["q_mvar", "p_mw", "vn_kv", "step", "max_step", "in_service", "res_p_mw", "res_q_mvar", "res_vm_pu"],
        "ext_grid": ["in_service", "va_degree", "vm_pu", "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar",
            "slack_weight", "res_p_mw", "res_q_mvar"],
        "line": ["c_nf_per_km", "df", "g_us_per_km", "in_service", "length_km", "max_i_ka", "max_loading_percent",
            "parallel", "r_ohm_per_km", "x_ohm_per_km", "res_p_from_mw", "res_q_from_mvar", "res_p_to_mw",
            "res_q_to_mvar", "res_pl_mw", "res_ql_mvar", "res_i_from_ka", "res_i_to_ka", "res_i_ka", "res_vm_from_pu",
            "res_va_from_degree", "res_vm_to_pu", "res_va_to_degree", "res_loading_percent"],
        "trafo": ["df", "i0_percent", "in_service", "max_loading_percent", "parallel", "pfe_kw", "shift_degree",
            "sn_mva", "tap_max", "tap_neutral", "tap_min", "tap_phase_shifter", "tap_pos", "tap_side",
            "tap_step_degree", "tap_step_percent", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "res_p_hv_mw",
            "res_q_hv_mvar", "res_p_lv_mw", "res_q_lv_mvar", "res_pl_mw", "res_ql_mvar", "res_i_hv_ka", "res_i_lv_ka",
            "res_vm_hv_pu", "res_va_hv_degree", "res_vm_lv_pu", "res_va_lv_degree", "res_loading_percent"],
    }

    def __init__(self):
        """Initializes a PandaPower backend."""
        super().__init__()

    def load_power_grid(self, file_path):
        """Loads a pandapower power grid instance from `.json`, `.pkl`, `.m` or `.mat`.

        Object names may be stored in a companion file that has the same name with the extension `.name`.
        Overrides the abstract `load_power_grid` method.
        """
        if file_path.endswith('.json'):
            power_grid = pp.from_json(file_path)
        elif file_path.endswith('.pkl'):
            power_grid = pp.from_pickle(file_path)
        elif file_path.endswith('.m'):
            power_grid = from_mpc(file_path, format='.m')
        elif file_path.endswith('.mat'):
            power_grid = from_mpc(file_path, format='.mat')
        else:
            raise NotImplementedError('No support for extension of file {}.'.format(file_path))
        power_grid.name = os.path.splitext(os.path.basename(file_path))[0]
        return power_grid

    def save_power_grid(self, power_grid, path, format='.json'):
        """Saves a power grid instance using the same name as in the initial file.

        TODO : on veut pouvoir modifier le format de sauvegarde du power_grid.

        Useful for saving a version of a test set modified by a trained neural network.
        Overrides the abstract `save_power_grid` method.
        """
        file_path = os.path.join(path, power_grid.name) + format
        if format=='.json':
            pp.to_json(power_grid, file_path)
        elif format=='.pkl':
            pp.to_pickle(power_grid, file_path)
        elif format in ['.m', '.mat']:
            to_mpc(power_grid, file_path, format=format)
        else:
            raise NotImplementedError('No support for extension of file {}'.format(file_path))

    @staticmethod
    def set_data_power_grid(power_grid, y):
        """Updates a power grid by setting features according to `y`.

        Overrides the abstract `set_data_power_grid` method.
        """
        for k in y.keys():
            for f in y[k].keys():
                try:
                    power_grid[k][f] = y[k][f]
                except ValueError:
                    print('Object class {} and feature {} are not available with PandaPower'.format(k, f))

    @staticmethod
    def run_power_grid(power_grid, **kwargs):
        """Runs a power flow simulation.

        Pandapower `runpp` options can be passed as keyword arguments.
        Overrides the abstract `run_power_grid` method.
        """
        try:
            pp.runpp(power_grid, **kwargs)
        except pp.powerflow.LoadflowNotConverged:
            pass

    @staticmethod
    def get_data_power_grid(power_grid, feature_names=None, address_names=None,
                            address_to_int=True, return_n_unique_addresses=False):
        """Extracts features from a pandapower network.

        If `address_to_int` is True, addresses are converted into unique integers that start at 0.
        If `initialize_latent_variables` is True, latent variables are initialized for each address.
        Overrides the abstract `get_data_network` method.
        """
        if feature_names is None:
            feature_names = dict()
        if address_names is None:
            address_names = dict()

        object_names = list(set(list(feature_names.keys()) + list(address_names.keys())))
        x = {}
        for key in object_names:
            if key == 'global':
                x[key] = {}
                table = pd.DataFrame({
                    'converged': [power_grid.converged * 1.], 'f_hz': [power_grid.f_hz * 1.], 'sn_mva': [power_grid.sn_mva * 1.]})
                for feature_name in feature_names[key]:
                    x[key][feature_name] = table[feature_name].astype(float).values
            else:
                x[key] = {}
                table = power_grid.get(key)
                res_table = power_grid.get('res_' + key)
                for feature_name in feature_names.get(key, []):
                    if feature_name == 'tap_side':
                        x[key][feature_name] = table[feature_name].map({'hv': 0., 'lv': 1.}).astype(float).values
                    elif feature_name[:4] == 'res_':
                        x[key][feature_name] = res_table[feature_name[4:]].astype(float).values
                    else:
                        x[key][feature_name] = table[feature_name].astype(float).values
                    x[key][feature_name] = np.nan_to_num(x[key][feature_name], copy=False) * 1

                for address_name in address_names.get(key, []):
                    if address_name == 'id':
                        x[key][address_name] = (key + '_' + table.index.astype(str)).values
                    elif address_name == 'bus_id':
                        x[key][address_name] = ('bus_' + table.bus.astype(str)).values
                    elif address_name == 'from_bus_id':
                        x[key][address_name] = ('bus_' + table.from_bus.astype(str)).values
                    elif address_name == 'to_bus_id':
                        x[key][address_name] = ('bus_' + table.to_bus.astype(str)).values
                    elif address_name == 'hv_bus_id':
                        x[key][address_name] = ('bus_' + table.hv_bus.astype(str)).values
                    elif address_name == 'lv_bus_id':
                        x[key][address_name] = ('bus_' + table.lv_bus.astype(str)).values
                    elif address_name == 'element':
                        x[key][address_name] = table.et.astype(str) + '_' + table.element.astype(str)
                    else:
                        x[key][address_name] = table[address_name].values.astype(str)
        clean_dict(x)
        if address_to_int:
            n_unique_addresses = convert_addresses_to_integers(x, address_names)

        if return_n_unique_addresses:
            return x, n_unique_addresses
        else:
            return x
