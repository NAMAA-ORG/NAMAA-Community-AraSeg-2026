import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np


torch = types.ModuleType("torch")
torch.cuda = SimpleNamespace(is_available=lambda: False)
torch.utils = SimpleNamespace(data=SimpleNamespace(DataLoader=None))

transformers = types.ModuleType("transformers")
transformers.AutoTokenizer = object

train = types.ModuleType("train")
train.Config = object

data_utils = types.ModuleType("data_utils")
data_utils.AraSegDataset = object
data_utils.make_collate = lambda _: None

model = types.ModuleType("model")
model.build_model = model.load_checkpoint = lambda *args, **kwargs: None

from scoring import _doc_f1, _macro

metrics = types.ModuleType("metrics")
metrics.collect_doc_probs = lambda *args, **kwargs: None
metrics.macro_f1_at = lambda docs, threshold: _macro([
    _doc_f1((prob >= threshold).astype(np.int64), gold)
    for prob, gold in docs.values()
])


def _tune_threshold(docs, grid=None):
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid
    return max(
        ((float(t), metrics.macro_f1_at(docs, float(t))) for t in grid),
        key=lambda item: item[1]["f1"],
    )


metrics.tune_threshold = _tune_threshold

utils = types.ModuleType("utils")
utils.resolve_model_path = lambda path: path

_STUBS = {
    "torch": torch,
    "transformers": transformers,
    "train": train,
    "data_utils": data_utils,
    "model": model,
    "metrics": metrics,
    "utils": utils,
}
_PREVIOUS = {name: sys.modules.get(name) for name in _STUBS}
sys.modules.update(_STUBS)
from ensemble import _exhaustive_sweep, _load_cache_member
for name, previous in _PREVIOUS.items():
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


GOLD = np.array([1, 0, 1, 0, 1, 0, 0, 1])
PROBS = [
    [0.33, 0.26, 0.62, 0.54, 0.34, 0.16, 0.53, 0.34],
    [0.73, 0.85, 0.50, 0.77, 0.71, 0.11, 0.14, 0.91],
    [0.79, 0.23, 0.95, 0.13, 0.27, 0.25, 0.09, 0.69],
]


def test_exhaustive_uses_requested_logit_combiner(tmp_path):
    members = [{"doc": (np.array(p), GOLD)} for p in PROBS]
    args = SimpleNamespace(combine="logit", weights=None, exhaustive_top_k=0)

    selected = _exhaustive_sweep(
        ["m0", "m1", "m2"], "PA", members, members, args, tmp_path
    )

    assert selected == [0, 2]


def test_named_cache_member_uses_shared_cache_convention(tmp_path):
    expected = {"doc": (np.array([0.2, 0.8]), np.array([0, 1]))}
    with (tmp_path / "PA_dev_foldbag_e32.pkl").open("wb") as fh:
        pickle.dump(expected, fh)

    actual = _load_cache_member(tmp_path, "PA", "dev", "foldbag_e32")

    np.testing.assert_array_equal(actual["doc"][0], expected["doc"][0])
