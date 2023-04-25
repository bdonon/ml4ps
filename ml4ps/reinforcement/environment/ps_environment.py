from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple

from gymnasium import Env, spaces
from ml4ps.h2mg import H2MG, H2MGStructure, H2MGSpace
import os


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
    # data_dir: str
    # backend: Any
    # empty_control_structure: H2MGStructure
    # empty_observation_structure: H2MGStructure


    # observation_structure: H2MGStructure
    # observation_space: H2MGSpace
    # action_structure: H2MGStructure
    # action_space: H2MGSpace
    # control_variable_structure: H2MGStructure
    #

    # action_space: spaces.Space
    # address_names: Dict
    # backend: Any
    # ctrl_var_names: Dict
    # data_dir: str
    # obs_feature_names: Dict
    # observation_space: spaces.Space
    # state: Any

    def __init__(self, data_dir):
        """
            Initialize environment
        """
        super().__init__()
        self.data_dir = data_dir

        path = os.path.join(data_dir, self.__class__.__name__)
        if not os.path.isdir(path):
            os.mkdir(path)

        # Comment on récupère les structures ...
        observation_structure_name = os.path.join(path, 'observation_structure.pkl')
        self.observation_structure = self.backend.get_max_structure(observation_structure_name,
                                                                    self.data_dir, self.empty_observation_structure)
        self.observation_space = H2MGSpace.from_structure(self.observation_structure)

        control_structure_name = os.path.join(path, 'control_structure.pkl')
        self.control_structure = self.backend.get_max_structure(control_structure_name,
                                                                self.data_dir, self.empty_control_structure)
        self.action_space = self._build_action_space(self.control_structure)

    @property
    @abstractmethod
    def backend(self):
        pass

    @property
    @abstractmethod
    def empty_observation_structure(self):
        pass

    @property
    @abstractmethod
    def empty_control_structure(self):
        pass

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[H2MG, Dict]:
        """Reset environement to a new power_grid for the given random seed."""
        super().reset(seed=seed, options=options)

    @abstractmethod
    def step(self, a) -> Tuple[H2MG, float, bool, bool, Dict]:
        """Update state with action and return new observation and reward."""
        pass

    @abstractmethod
    def get_information(self, state: Any) -> Dict:
        """extract features from power grid power_grid and return observation dict."""
        pass

    # @abstractmethod
    # def build_observation_space(self, data_dir, observation_space: spaces.Space = None) -> Dict:
    #     """Return observation space and corresponding n_obj."""
    #     pass


    @abstractmethod
    def _build_action_space(self, control_structure):
        pass