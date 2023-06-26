from typing import Any, Dict
from ml4ps.h2mg import H2MG

class BaseNN:
    def apply(self, params: Dict, h2mg_in: H2MG, **kwargs):
        pass