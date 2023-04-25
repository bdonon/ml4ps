import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import jax.numpy as jnp
import mlflow
import numpy as np
from omegaconf import DictConfig, ListConfig


def dict_mean(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: np.nanmean(v) for (k, v) in d.items()}


def process_venv_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        d = _remove_underscore_keys(d)
        d = {prefix + k: v for (k, v) in d.items()}
        d = _handle_venv_finals( d, prefix)
        return  dict_mean(d)

def _remove_underscore_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    keys_to_remove = []
    for k in d.keys():
        if k.startswith('_'):
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del d[k]
    return d
        
def _handle_venv_finals(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    keys_to_remove = []
    final_info = {}
    for k, v  in d.items():
        if k.endswith("final_observation"):
            keys_to_remove.append(k)
        elif k.endswith("final_info"):
            keys_to_remove.append(k)
            final_info = _get_final_info_dict(v, prefix)
    for k in keys_to_remove:
        del d[k]
    return {**d, **final_info}

def _get_final_info_dict(infos: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    res = {}
    not_none_infos = [info for info in infos if info is not None]
    for key in not_none_infos[0]:
        # TODO: check
        try:
            res[prefix+"final_"+key] = np.nanmean([info[key] for info in not_none_infos])
        except:
            continue
    return res

class BaseLogger(ABC):
    def log_hyperparams(params):
        pass
    
    def log_metrics(metrics: Dict[str, Any], step: int=None) -> None:
        pass
    
    @abstractmethod
    def log_metrics_dict(metrics: Dict[str, Any], step: int=None) -> None:
        pass
    
    def log_dicts(self, step, prefix, *dicts):
        for d in dicts:
            d = process_venv_dict(d, prefix)
            self.log_metrics_dict(d, step)
    
    def finalize(self):
        pass

class TensorboardLogger(BaseLogger):
    pass

class CSVLogger(BaseLogger):
    def log_hyperparams(self, params):
        pass
    
    def log_metrics(self, metrics, step=None):
        pass
    
    def log_metrics_dict(self, metrics: Dict, step=None):
        pass

class MLFlowLogger(BaseLogger):
    def __init__(self, experiment_name, run_name, res_dir=None) -> None:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp:
            experiment_id = exp.experiment_id
        else:
            experiment_id = mlflow.create_experiment(name=experiment_name)
        mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        if res_dir is not None:
            mlflow.set_tracking_uri(res_dir)
    
    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_metrics_dict(self, metrics: Dict, step=None):
        for key, value in metrics.items():
            v = np.asarray(value)
            mlflow.log_metric(key, v, step=step)
    
    def finalize(self):
        return mlflow.end_run()
