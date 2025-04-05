from typing import Any
import copy
import json


class ModelConfig:

    model_name: str = ""

    def __init__(self):
        pass

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(self.__dict__)

        if hasattr(self.__class__, "model_name"):
            output["model_name"] = self.__class__.model_name

        return output

    def to_json_string(self) -> str:
        config_dict = self.to_dict()

        return json.dumps(config_dict, indent=2, sort_keys=True)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"
