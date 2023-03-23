import pandapower.converter as pc
import pandapower as pp
import pandas as pd
import numpy as np
import mat73
import json
import copy
import os

from pandapower.converter.matpower.from_mpc import _adjust_ppc_indices, _change_ppc_TAP_value
from pandapower.converter.matpower.to_mpc import _ppc2mpc
from scipy.io import savemat

from ml4ps.backend.interface import AbstractBackend
from ml4ps.h2mg import H2MG, H2MGStructure, HyperEdges, HyperEdgesStructure

MATPOWER_DIR = "matpower"
NAMES_DIR = "names"
SHUNTS_DIR = "shunts"

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
        #power_grid = pc.from_mpc(file_path)
        from pandapower.converter.matpower.from_mpc import _m2ppc, from_ppc
        ppc = _m2ppc(file_path, "mpc")
        mask = ppc["bus"][:, 1] == 1
        ppc["bus"][mask, 1] = 2
        ppc["bus"][:, 11] = 1.1
        ppc["bus"][:, 12] = 0.9
        #ppc["gen"][:, 1] = 1.
        #ppc["gen"][:, 7] = 1.
        from pandapower.converter.pypower.from_ppc import _gen_to_which
        is_ext_grid, is_gen, is_sgen = _gen_to_which(ppc)

        #print(ppc["gen"].shape)
        power_grid = from_ppc(ppc)
        return power_grid

    def load_object_names(power_grid, file_path):
        try:
            power_grid_name = os.path.splitext(os.path.basename(file_path))[0]
            names_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), NAMES_DIR)
            name_path = os.path.join(names_dir, power_grid_name+".json")
            with open(name_path, 'r') as f:
                name_dict = json.load(f)
            for key, name_list in name_dict.items():
                power_grid.get(key).name = name_list
        except:
            pass

    def load_shunts(power_grid, file_path):
        power_grid_name = os.path.splitext(os.path.basename(file_path))[0]
        shunts_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), SHUNTS_DIR)
        shunt_path = os.path.join(shunts_dir, power_grid_name+".csv")
        #name_path = os.path.splitext(file_path)[0] + '_shunt.csv'
        shunts = pd.read_csv(shunt_path)
        shunts.rename(columns={'Bs': 'q_mvar', 'Gs': 'p_mw', 'status': 'in_service'}, inplace=True)
        shunts["q_mvar"] = - shunts["q_mvar"]  # Bs = - Q

        # ["bus", "q_mvar", "p_mw", "vn_kv", "max_step", "in_service"]
        shunts["step"] = 0.
        for i, shunt in shunts.iterrows():
            bus_id = shunt['bus']
            if bus_id in power_grid.shunt.bus.values:
                q_mvar = power_grid.shunt.q_mvar[power_grid.shunt.bus == bus_id]
                step = q_mvar / shunt.q_mvar
                shunts["step"].iloc[i] = step

        shunts["name"] = "0"
        power_grid.shunt = shunts

    if format=='.mat':
        try:
            power_grid = pc.from_mpc(file_path)
        except NotImplementedError:
            power_grid = from_mpc73(file_path)
    elif format == '.m':
        power_grid = from_m(file_path)
    load_shunts(power_grid, file_path)
    load_object_names(power_grid, file_path)
    return power_grid


def to_mpc(net, path, format='.mat', **kwargs):
    """Modification of the `to_mpc` implementation of pandapower
    (https://github.com/e2nIEE/pandapower/blob/develop/pandapower/converter/matpower/to_mpc.py)

    The present implementation saves all objects and sets the status of out-of-service
    objects to 0.
    The default implementation deletes objects that are out-of-service, which
    completely alters the object ordering. For visualization purpose, panoramix relies
    heavily on this ordering.
    """


    if not os.path.exists(os.path.join(os.path.dirname(path), MATPOWER_DIR)):
        os.mkdir(os.path.join(os.path.dirname(path), MATPOWER_DIR))
    if not os.path.exists(os.path.join(os.path.dirname(path), NAMES_DIR)):
        os.mkdir(os.path.join(os.path.dirname(path), NAMES_DIR))
    if not os.path.exists(os.path.join(os.path.dirname(path), SHUNTS_DIR)):
        os.mkdir(os.path.join(os.path.dirname(path), SHUNTS_DIR))


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
    ppc = pp.converter.to_ppc(net, take_slack_vm_limits=False)

    # Manually change the Gen and Branch status to reflect the actual in_service values
    ppc['gen'][:, 7] = ppc_gen_status
    ppc['branch'][:, 10] = ppc_branch_status

    # Get the current step and max step for shunts
    shunts = net.shunt[["bus", "q_mvar", "p_mw", "vn_kv", "max_step", "in_service"]]#.astype(float).values

    shunts["q_mvar"] = - shunts["q_mvar"] # Bs = - Q
    shunts.rename(columns={'q_mvar': 'Bs', 'p_mw': 'Gs', 'in_service': 'status'}, inplace=True)
    #ppc['bus'][:, 4] = 0. # Delete shunts because they are stored in shunt_steps
    #ppc['bus'][:, 5] = 0.

    # Untouched part
    mpc = dict()
    mpc["mpc"] = _ppc2mpc(ppc)
    if format == '.mat':
        savemat(filepath, mpc)
    elif format == '.m':

        filepath = os.path.join(os.path.join(os.path.dirname(path), MATPOWER_DIR), net.name + ".m")

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
            #f.write("%\tbus_i\tBS\tGs\tvn_kv\tstep\tmax_step\tstatus\n")
            #f.write("mpc.shunt_steps = [\n")
            #write_table(f, shunt_steps)
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

    names_path = os.path.join(os.path.join(os.path.dirname(path), NAMES_DIR), net.name + ".json")
    with open(names_path, "w") as outfile:
        json.dump(names, outfile)

    shunts_path = os.path.join(os.path.join(os.path.dirname(path), SHUNTS_DIR), net.name + ".csv")
    shunts.to_csv(shunts_path, index=False)

    return mpc


#
# def from_mpc(file_path, format='.mat'):
#     """Implementation of from_mpc that supports both .m and .mat files, while allowing matlab 7.3 .m format."""
#
#     def from_mpc73(mpc_file, f_hz=50, validate_conversion=False, **kwargs):
#         """Implementation of from_mpc that supports matlab 7.3 .m format."""
#         mpc = mat73.loadmat(mpc_file)
#         ppc = {"version": "X.X", "baseMVA": mpc["mpc"]["baseMVA"], "bus": mpc["mpc"]["baseMVA"],
#             "gen": mpc["mpc"]["gen"], "branch": mpc["mpc"]["branch"], "gencost": mpc["mpc"]["gencost"]}
#         _adjust_ppc_indices(ppc)
#         _change_ppc_TAP_value(ppc)
#         return pc.from_ppc(ppc, f_hz=f_hz, validate_conversion=validate_conversion, **kwargs)
#
#     def from_m(file_path):
#         power_grid = pc.from_mpc(file_path)
#         pattern = r'mpc\.{}\s*=\s*\[[\n]?(?P<data>.*?)[\n]?\];'.format("shunt_steps")
#         with open(file_path) as f:
#             match = re.search(pattern, f.read(), re.DOTALL)
#         match = match.groupdict().get('data', None)
#         match = match.strip("'").strip('"')
#         _list = list()
#         for line in match.splitlines():
#             line = line.split('%')[0]
#             line = line.replace(';', '')
#             if line.strip():
#                 _list.append([int_else_float_except_string(s) for s in line.strip().split()])
#         power_grid.shunt = pd.DataFrame(_list,
#                                         columns=["bus", "q_mvar", "p_mw", "vn_kv", "step", "max_step", "in_service"])
#         power_grid.shunt["name"] = ""
#         return power_grid
#
#     def load_object_names(power_grid, file_path):
#         try:
#             name_path = os.path.splitext(file_path)[0] + '.name'
#             with open(name_path, 'r') as f:
#                 name_dict = json.load(f)
#             for key, name_list in name_dict.items():
#                 power_grid.get(key).name = name_list
#         except:
#             pass
#
#     if format=='.mat':
#         try:
#             power_grid = pc.from_mpc(file_path)
#         except NotImplementedError:
#             power_grid = from_mpc73(file_path)
#     elif format == '.m':
#         power_grid = from_m(file_path)
#     load_object_names(power_grid, file_path)
#     return power_grid
#
#
# def to_mpc(net, filepath, format='.mat', **kwargs):
#     """Modification of the `to_mpc` implementation of pandapower
#     (https://github.com/e2nIEE/pandapower/blob/develop/pandapower/converter/matpower/to_mpc.py)
#
#     The present implementation saves all objects and sets the status of out-of-service
#     objects to 0.
#     The default implementation deletes objects that are out-of-service, which
#     completely alters the object ordering. For visualization purpose, panoramix relies
#     heavily on this ordering.
#     """
#
#     net = copy.deepcopy(net)
#
#     # Save actual object status
#     gen_status = net.gen.in_service.astype(float).values
#     ext_grid_status = net.ext_grid.in_service.astype(float).values
#     line_status = net.line.in_service.astype(float).values
#     trafo_status = net.trafo.in_service.astype(float).values
#     ppc_gen_status = np.concatenate([ext_grid_status, gen_status])
#     ppc_branch_status = np.concatenate([line_status, trafo_status])
#
#     # Set all objects to be in_service and convert to pypower object
#     net.gen.in_service = True
#     net.ext_grid.in_service = True
#     net.line.in_service = True
#     net.trafo.in_service = True
#     ppc = pp.converter.to_ppc(net, **kwargs)
#
#     # Manually change the Gen and Branch status to reflect the actual in_service values
#     ppc['gen'][:, 7] = ppc_gen_status
#     ppc['branch'][:, 10] = ppc_branch_status
#
#     # Get the current step and max step for shunts
#     shunt = net.shunt[["bus", "q_mvar", "p_mw", "vn_kv", "step", "max_step", "in_service"]].astype(float).values
#     ppc['bus'][:, 4] = 0.
#     ppc['bus'][:, 5] = 0.
#
#     # Untouched part
#     mpc = dict()
#     mpc["mpc"] = _ppc2mpc(ppc)
#     if format == '.mat':
#         savemat(filepath, mpc)
#     elif format == '.m':
#
#         def write_table(f, arr, max_col=None):
#             for r in arr:
#                 for v in r[:max_col]:
#                     if v.is_integer():
#                         f.write("\t{}".format(v.astype(int)))
#                     else:
#                         f.write("\t{:.6f}".format(v))
#                 f.write(";\n")
#             f.write("];\n")
#
#         with open(filepath, "w") as f:
#             f.write("function mpc = powergrid\n")
#             f.write("mpc.version = '2';\n")
#             f.write("mpc.baseMVA = 100;\n")
#             f.write("mpc.bus = [\n")
#             write_table(f, mpc["mpc"]["bus"], max_col=13)
#             f.write("mpc.gen = [\n")
#             write_table(f, mpc["mpc"]["gen"], max_col=21)
#             f.write("mpc.branch = [\n")
#             write_table(f, mpc["mpc"]["branch"], max_col=13)
#             f.write("%\tbus_i\tQ\tP\tvn_kv\tstep\tmax_step\tstatus\n")
#             f.write("mpc.shunt_steps = [\n")
#             write_table(f, shunt)
#             f.close()
#
#
#     # Save names
#     names = {
#         'bus': list(net.bus.name.astype(str).values),
#         'gen': list(net.gen.name.astype(str).values),
#         'load': list(net.load.name.astype(str).values),
#         'line': list(net.line.name.astype(str).values),
#         'trafo': list(net.trafo.name.astype(str).values),
#         'ext_grid': list(net.ext_grid.name.astype(str).values),
#         'shunt': list(net.shunt.name.astype(str).values),
#     }
#     with open(os.path.splitext(filepath)[0] + '.name', "w") as outfile:
#         json.dump(names, outfile)
#
#     return mpc


class PandaPowerBackend(AbstractBackend):
    """Backend implementation that uses `PandaPower <http://www.pandapower.org>`_."""

    valid_extensions = (".json", ".pkl", ".mat")
    valid_structure = H2MGStructure()

    bus_structure = HyperEdgesStructure(addresses=["id", "name"], features=["in_service", "max_vm_pu", 'min_vm_pu',
        "vn_kv", "res_vm_pu", "res_va_degree", "res_p_mw", "res_q_mvar"])
    valid_structure.add_local_hyper_edges_structure("bus", bus_structure)

    load_structure = HyperEdgesStructure(addresses=["bus_id", "name"], features=["const_i_percent", "const_z_percent",
        "controllable", "in_service", "p_mw", "q_mvar", "scaling", "sn_mva", "res_p_mw", "res_q_mvar"])
    valid_structure.add_local_hyper_edges_structure("load", load_structure)

    sgen_structure = HyperEdgesStructure(addresses=["bus_id", "name"], features=["in_service", "p_mw",
        "q_mvar", "scaling", "sn_mva", "current_source", "res_p_mw", "res_q_mvar"])
    valid_structure.add_local_hyper_edges_structure("sgen", sgen_structure)

    gen_structure = HyperEdgesStructure(addresses=["bus_id", "name"], features=["controllable", "in_service", "p_mw",
        "scaling", "sn_mva", "vm_pu", "slack", "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "slack_weight",
        "res_p_mw", "res_q_mvar", "res_va_degree", "res_vm_pu"])
    valid_structure.add_local_hyper_edges_structure("gen", gen_structure)

    shunt_structure = HyperEdgesStructure(addresses=["bus_id", "name"], features=["q_mvar", "p_mw", "vn_kv", "step",
        "max_step", "in_service", "res_p_mw", "res_q_mvar", "res_vm_pu"])
    valid_structure.add_local_hyper_edges_structure("shunt", shunt_structure)

    ext_grid_structure = HyperEdgesStructure(addresses=["bus_id", "name"], features=["in_service", "va_degree", "vm_pu",
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "slack_weight", "res_p_mw", "res_q_mvar"])
    valid_structure.add_local_hyper_edges_structure("ext_grid", ext_grid_structure)

    line_structure = HyperEdgesStructure(addresses=["from_bus_id", "to_bus_id", "name"], features=["c_nf_per_km", "df",
        "g_us_per_km", "in_service", "length_km", "max_i_ka", "max_loading_percent", "parallel", "r_ohm_per_km",
        "x_ohm_per_km", "res_p_from_mw", "res_q_from_mvar", "res_p_to_mw", "res_q_to_mvar", "res_pl_mw", "res_ql_mvar",
        "res_i_from_ka", "res_i_to_ka", "res_i_ka", "res_vm_from_pu", "res_va_from_degree", "res_vm_to_pu",
        "res_va_to_degree", "res_loading_percent"])
    valid_structure.add_local_hyper_edges_structure("line", line_structure)

    trafo_structure = HyperEdgesStructure(addresses=["hv_bus_id", "lv_bus_id", "name"], features=["df", "i0_percent",
        "in_service", "max_loading_percent", "parallel", "pfe_kw", "shift_degree", "sn_mva", "tap_max", "tap_neutral",
        "tap_min", "tap_phase_shifter", "tap_pos", "tap_side", "tap_step_degree", "tap_step_percent", "vn_hv_kv",
        "vn_lv_kv", "vk_percent", "vkr_percent", "res_p_hv_mw", "res_q_hv_mvar", "res_p_lv_mw", "res_q_lv_mvar",
        "res_pl_mw", "res_ql_mvar", "res_i_hv_ka", "res_i_lv_ka", "res_vm_hv_pu", "res_va_hv_degree", "res_vm_lv_pu",
        "res_va_lv_degree", "res_loading_percent"])
    valid_structure.add_local_hyper_edges_structure("trafo", trafo_structure)

    global_structure = HyperEdgesStructure(features=["converged", "f_hz", "sn_mva"])
    valid_structure.add_global_hyper_edges_structure(global_structure)

    def __init__(self, default_structure=None):
        """Initializes a PandaPower backend."""
        self.default_structure = default_structure
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
    def set_h2mg_into_power_grid(power_grid, h2mg):
        """Updates a power grid by setting features according to `h2mg`.

        Overrides the abstract `set_data_power_grid` method.
        """
        for k, hyper_edges in h2mg.local_hyper_edges.items():
            for f, v in hyper_edges.features.items():
                #try:
                #print(power_grid[k][f], v)
                power_grid[k][f] = v
                #except ValueError:
                #    print('Object class {} and feature {} are not available with PandaPower'.format(k, f))

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

    def get_h2mg_from_power_grid(self, power_grid, structure=None, str_to_int=True):
        """Extracts features from a pandapower network.

        Overrides the abstract `get_data_network` method.
        """
        if structure is not None:
            current_structure = structure
        elif self.default_structure is not None:
            current_structure = self.default_structure
        else:
            current_structure = self.valid_structure

        h2mg = H2MG()

        if current_structure.global_hyper_edges_structure is not None:
            global_features_dict = {}
            for k in current_structure.global_hyper_edges_structure.features:
                if k == 'converged':
                    global_features_dict[k] = np.array([power_grid.converged * 1.])
                elif k == 'f_hz':
                    global_features_dict[k] = np.array([power_grid.f_hz * 1.])
                elif k == 'sn_mva':
                    global_features_dict[k] = np.array([power_grid.sn_mva * 1.])
                else:
                    raise NotImplementedError
            h2mg.add_global_hyper_edges(HyperEdges(features=global_features_dict))

        if current_structure.local_hyper_edges_structure is not None:
            for k, hyper_edges_structure in current_structure.local_hyper_edges_structure.items():
                table = _get_local_table(power_grid, k)
                if not table.empty:

                    addresses_dict = None
                    if hyper_edges_structure.addresses is not None:
                        addresses_dict = {}
                        for address_name in hyper_edges_structure.addresses:
                            addresses_dict[address_name] = table[address_name].astype(str).values

                    features_dict = None
                    if hyper_edges_structure.features is not None:
                        features_dict = {}
                        for feature_name in hyper_edges_structure.features:
                            features_dict[feature_name] = np.nan_to_num(table[feature_name].astype(float).values, copy=False) * 1

                    hyper_edges = HyperEdges(features=features_dict, addresses=addresses_dict)
                    if not hyper_edges.is_empty():
                        h2mg.add_local_hyper_edges(k, HyperEdges(features=features_dict, addresses=addresses_dict))

        if str_to_int:
            h2mg.convert_str_to_int()
        return h2mg

    def get_valid_files(self, path, shuffle=False, n_samples=None):
        """Gets file that have a valid extension w.r.t. the backend, from path."""
        files = []
        for f in sorted(os.listdir(path)):
            if f.endswith(".json"):
                files.append(os.path.join(path, f))
        if (MATPOWER_DIR in os.listdir(path)) and (NAMES_DIR in os.listdir(path)) and (SHUNTS_DIR in os.listdir(path)):
            for f in sorted(os.listdir(os.path.join(path, MATPOWER_DIR))):
                if f.endswith(".m"):
                    files.append(os.path.join(os.path.join(path, MATPOWER_DIR), f))
        if not files:
            raise FileNotFoundError("There is no valid file in {}".format(path))
        if shuffle:
            np.random.shuffle(files)
        if n_samples is not None:
            return files[:n_samples]
        else:
            return files


def _get_local_table(power_grid, k):

    table = power_grid.get(k)
    res_table = power_grid.get('res_' + k).add_prefix('res_')
    table = pd.concat([table, res_table], axis=1)

    table["id"] = k + "_" + table.index.astype(str)

    if "tap_side" in table.columns:
        table["tap_side"] = table["tap_side"].map({'hv': 0., 'lv': 1.})
    if "bus" in table.columns:
        table["bus_id"] = 'bus_' + table.bus.astype(str)
    if "from_bus" in table.columns:
        table["from_bus_id"] = 'bus_' + table.from_bus.astype(str)
    if "to_bus" in table.columns:
        table["to_bus_id"] = 'bus_' + table.to_bus.astype(str)
    if "hv_bus" in table.columns:
        table["hv_bus_id"] = 'bus_' + table.hv_bus.astype(str)
    if "lv_bus" in table.columns:
        table["lv_bus_id"] = 'bus_' + table.lv_bus.astype(str)
    if ("element" in table.columns) and ("et" in table.columns):
        table["element_id"] = table.et.astype(str) + '_' + table.element.astype(str)

    return table