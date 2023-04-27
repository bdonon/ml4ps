from gymnasium.envs.registration import register
from ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV1 import \
    VoltageManagementPandapowerV1
from ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV2 import \
    VoltageManagementPandapowerV2
from ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV3 import \
    VoltageManagementPandapowerV3

register(
    id="VoltageManagementPandapowerV1",
    entry_point="ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV1:VoltageManagementPandapowerV1",
)

register(
    id="VoltageManagementPandapowerV2",
    entry_point="ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV2:VoltageManagementPandapowerV2",
)

register(
    id="VoltageManagementPandapowerV3",
    entry_point="ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV3:VoltageManagementPandapowerV3",
)
