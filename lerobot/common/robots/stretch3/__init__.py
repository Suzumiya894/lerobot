from .configuration_stretch3 import Stretch3RobotConfig
# from .robot_stretch3 import Stretch3Robot
if Stretch3RobotConfig.is_remote_server:
    from .stretch_server import StretchRobotServer as MyStretchRobot
else:
    from .mystretch import MyStretchRobot

