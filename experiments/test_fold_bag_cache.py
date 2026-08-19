import pickle
import sys
import types

import numpy as np


train = types.ModuleType("train")
train.Config = object

ensemble = types.ModuleType("ensemble")
ensemble._member_probs = lambda *args, **kwargs: None

metrics = types.ModuleType("metrics")
metrics.tune_threshold = metrics.macro_f1_at = lambda *args, **kwargs: None

_STUBS = {"train": train, "ensemble": ensemble, "metrics": metrics}
_PREVIOUS = {name: sys.modules.get(name) for name in _STUBS}
sys.modules.update(_STUBS)
from fold_bag import write_cache
for name, previous in _PREVIOUS.items():
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def test_write_cache_uses_shared_member_filename(tmp_path):
    docs = {"d": (np.array([0.1, 0.9]), np.array([0, 1]))}

    path = write_cache(tmp_path, "PA", "dev", "foldbag_e32", docs)

    assert path.name == "PA_dev_foldbag_e32.pkl"
    with path.open("rb") as fh:
        loaded = pickle.load(fh)
    np.testing.assert_array_equal(loaded["d"][0], docs["d"][0])
