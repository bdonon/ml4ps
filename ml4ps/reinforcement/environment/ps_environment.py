from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple

from gymnasium import Env, spaces
from ml4ps.h2mg import H2MG


class PSBaseEnv(Env, ABC):
    """
        Power system base environment.

        Attributes:
            action_space: spaces.Space
            address_names: Dict[List]
            backend: Any
            ctrl_var_names: Dict[List]
            data_dir: str
            obs_feature_names: Dict[List]
            observation_space: spaces.Space
            state: Any
    """

    action_space: spaces.Space
    address_names: Dict
    backend: Any
    ctrl_var_names: Dict
    data_dir: str
    obs_feature_names: Dict
    observation_space: spaces.Space
    state: Any

    def __init__(self):
        """
            Initialize environment
        """
        super().__init__()

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[H2MG, Dict]:
        """Reset environement to a new power_grid for the given random seed."""
        pass

    @abstractmethod
    def step(self, a) -> Tuple[H2MG, float, bool, bool, Dict]:
        """Update state with action and return new observation and reward."""
        pass

    @abstractmethod
    def get_information(self, state: Any) -> Dict:
        """extract features from power grid power_grid and return observation dict."""
        pass

    @abstractmethod
    def build_observation_space(self, data_dir, observation_space: spaces.Space = None) -> Dict:
        """Return observation space and corresponding n_obj."""
        pass
