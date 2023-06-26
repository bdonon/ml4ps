import os
from collections import deque
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from gymnasium import spaces
from ml4ps.h2mg import H2MG
from ml4ps.reinforcement.environment import PSBaseEnv


class TestEnv(PSBaseEnv):
    def __init__(self, env: PSBaseEnv, save_folder: str = None, auto_reset=True, max_steps=None) -> 'TestEnv':
        self.env = deepcopy(env)
        self.auto_reset = auto_reset
        self.max_steps = max_steps
        self.current_step = 0
        self.filelist = self.env.backend.get_valid_files(self.env.data_dir)
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
        self.current_step += 1
        if self.max_steps is not None and self.current_step >= self.max_steps:
            terminated = True
            truncated = True
        if terminated:
            if self.save_folder is not None:
                self.env.backend.save_power_grid(
                    self.state.power_grid, path=self.save_folder)
            if len(self.filelist) <= 0:
                self._is_done = True
            if self.auto_reset:
                obs, info = self.reset(options={"load_new_power_grid": True})
                self.current_step = 0

        return obs, reward, terminated, truncated, info

    def get_information(self, state: Any, action=None, reward=None) -> Dict:
        return self.env.get_information(state, action, reward)

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

    @property
    def empty_info_structure(self):
        return self.env.empty_info_structure

    def __len__(self):
        return len(self.filelist)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def is_done(self) -> bool:
        return self._is_done
