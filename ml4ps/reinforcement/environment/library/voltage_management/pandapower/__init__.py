from gymnasium.envs.registration import register
from ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV1 import \
    VoltageManagementPandapowerV1
from ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV1B import \
    VoltageManagementPandapowerV1B
from ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV2 import \
    VoltageManagementPandapowerV2
from ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV2B import \
    VoltageManagementPandapowerV2B
from ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV3 import \
    VoltageManagementPandapowerV3
from ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV3B import \
    VoltageManagementPandapowerV3B


register(
    id="VoltageManagementPandapowerV1",
    entry_point="ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV1:VoltageManagementPandapowerV1",
)
register(
    id="VoltageManagementPandapowerV1B",
    entry_point="ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV1B:VoltageManagementPandapowerV1B",
)

register(
    id="VoltageManagementPandapowerV2",
    entry_point="ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV2:VoltageManagementPandapowerV2",
)

register(
    id="VoltageManagementPandapowerV2B",
    entry_point="ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV2B:VoltageManagementPandapowerV2B",
)

register(
    id="VoltageManagementPandapowerV3",
    entry_point="ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV3:VoltageManagementPandapowerV3",
)


register(
    id="VoltageManagementPandapowerV3B",
    entry_point="ml4ps.reinforcement.environment.library.voltage_management.pandapower.VoltageManagementPandapowerV3B:VoltageManagementPandapowerV3B",
)
