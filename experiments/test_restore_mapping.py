import numpy as np

from naqta_restore import NAQTA_MARKS, map_binary_back, map_probs_back, restore_doc
from cache_probs import map_restored_docs


owner = [0, 0, 1, 2, 2]
probs = np.asarray([.2, .8, .4, .55, .7], np.float32)

assert NAQTA_MARKS[2] == "،"
assert np.allclose(map_probs_back(probs, owner, 3), [.8, .4, .7])
for threshold in (.3, .6, .75):
    assert np.array_equal(
        map_probs_back(probs, owner, 3) >= threshold,
        map_binary_back(probs >= threshold, owner, 3),
    )
assert np.array_equal(map_binary_back([1, 0], [-1, 0], 1), [0])
assert np.allclose(map_probs_back([.9, .2], [-1, 0], 1), [.2])

restored = {"d": (np.array([0.2, 0.8, 0.4]), np.array([0, 0, 1]))}
mapped = map_restored_docs(restored, {"d": [0, 0, 1]}, {"d": [1, 0]})
probs, gold = mapped["d"]
assert np.allclose(probs, [0.8, 0.4])
assert gold.tolist() == [1, 0]

# naqta_comma is a per-call override, never a change to NAQTA_MARKS itself.
_p = np.zeros((1, 8), np.float32)
_p[0, 2] = 0.9  # class 2 (the comma) is the best non-O mark for this word
_toks_default, _, _ = restore_doc(["w0"], [0], _p, min_p=0.5)
_toks_native, _, _ = restore_doc(["w0"], [0], _p, min_p=0.5, comma="،")
_toks_ascii, _, _ = restore_doc(["w0"], [0], _p, min_p=0.5, comma=",")
assert _toks_default == ["w0", "،"] == _toks_native, _toks_default
assert _toks_ascii == ["w0", ","], _toks_ascii

print("restore mapping check OK")
