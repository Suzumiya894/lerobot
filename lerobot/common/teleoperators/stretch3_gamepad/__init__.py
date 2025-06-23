from .configuration_stretch3 import Stretch3GamePadConfig
# from .stretch3_gamepad import Stretch3GamePad
if Stretch3GamePadConfig.mock:
    from .stretch3_gamepad_mock import Stretch3GamePad
else:
    from .my_stretch3_gamepad import Stretch3GamePad
