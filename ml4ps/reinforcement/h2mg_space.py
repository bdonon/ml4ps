from gymnasium import spaces

class H2MGSpace(spaces.Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)