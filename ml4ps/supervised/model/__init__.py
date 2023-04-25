from ml4ps.supervised.model.base import *
from ml4ps.supervised.model.classifier import *
from ml4ps.supervised.model.regressor import *


def get_model(model_type: str, *model_args, **model_kwargs):
    if model_type == "regressor":
        return Regressor(*model_args, **model_kwargs)
    elif model_type == "classifier":
        return Classifier(*model_args, **model_kwargs)
    else:
        raise NotImplementedError("No existing model of type {}.".format(model_type))
