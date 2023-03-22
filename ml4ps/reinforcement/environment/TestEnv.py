from typing import Optional, Tuple, Dict, Any
from ml4ps.h2mg import H2MG
from ml4ps.reinforcement.environment import PSBaseEnv
from gymnasium import spaces
from collections import deque
import os

class TestEnv(PSBaseEnv):
    def __init__(self, env: PSBaseEnv, save_folder: str) -> 'TestEnv':
        self.env = env
        self.filelist = env.backend.get_valid_files(env.data_dir)
        self.filelist = deque(self.filelist)
        self.save_folder = save_folder
        self._maxlen = len(self.filelist)
        if self._maxlen < 1:
            raise ValueError("No power grid snapshot to load.")
        self._is_done = False
    
    def get_next_file(self):
        if len(self.filelist) <= 0:
            return None
        return self.filelist.popleft()
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[H2MG, Dict]:
        if options is None:
            options = {}
        file_path = self.get_next_file()
        if file_path is None:
            return None, {"no_more_files": True}
        options["power_grid_path"] = file_path
        options["load_new_power_grid"] = True
        return self.env.reset(seed=seed, options=options)
    
    def step(self, a) -> Tuple[H2MG, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(a)
        if terminated:
            self.env.backend.save_power_grid(self.state.power_grid, path=self.save_folder)
            if len(self.filelist) <= 0:
                self._is_done=True
            obs, info = self.reset()
            path = self.save_folder
            
            
        return obs, reward, terminated, truncated, info
    
    def get_information(self, state: Any) -> Dict:
        return self.env.get_information(state)

    def build_observation_space(self, data_dir, observation_space: spaces.Space = None) -> Dict:
        return self.env.build_observation_space(data_dir, observation_space)
    
    @property
    def action_space(self) -> spaces.Space:
        return self.env.action_space
    
    @property
    def observation_space(self) -> spaces.Space:
        return self.env.observation_space

    @property
    def backend(self):
        return self.env.backend
    @property
    def data_dir(self):
        return self.env.data_dir
    
    @property
    def state(self):
        return self.env.state
    
    def _build_action_space(self, control_structure) -> spaces.Space:
        return self.env._build_action_space(control_structure)
    
    @property
    def empty_control_structure(self):
        return self.env.empty_control_structure
    
    @property
    def empty_observation_structure(self):
        return self.env.empty_observation_structure
    
    def __len__(self):
        return len(self.filelist)
    
    @property
    def maxlen(self) -> int:
        return self._maxlen
    
    @property
    def is_done(self) -> bool:
        return self._is_done
    
    # address_names: Dict
    # ctrl_var_names: Dict
    # obs_feature_names: Dict