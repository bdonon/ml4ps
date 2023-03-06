from typing import Dict
import datetime
import mlflow
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    def log_hyperparams(params):
        pass
    
    def log_metrics(metrics, step=None):
        pass
    
    def log_metrics_dict(metrics: Dict, step=None):
        pass
    
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

class MLFlowLogger():
    def __init__(self, experiment_name) -> None:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp:
            experiment_id = exp.experiment_id
        else:
            experiment_id = mlflow.create_experiment(name=experiment_name)
        mlflow.start_run(experiment_id=experiment_id)
    
    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_metrics_dict(self, metrics: Dict, step=None):
        # Warning: removes nan !
        for key, value in metrics.items():
            if key.startswith('_'):
                continue
            if key.endswith("final_info"):
                self.log_final_info(value, step)
                continue
            if key.endswith("final_observation"):
                continue
            v = np.asarray(jnp.nanmean(value))
            mlflow.log_metric(key, v, step=step)    
    
    def finalize(self):
        return mlflow.end_run()