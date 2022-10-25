import json
from os.path import dirname

with open(dirname(__file__) + "/_params.json") as h:
    _params = json.load(h)

with open(dirname(__file__) + "/pkg_info.json") as h:
    _info = json.load(h)

__version__ = _info["version"]