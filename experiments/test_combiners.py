"""Assert-based checks for combiners.py. Run: python experiments/test_combiners.py"""
import numpy as np
from combiners import combine_prob, combine_logit, combine_rank, COMBINERS


def test_prob_is_plain_mean():
    a = np.array([0.2, 0.8], dtype=np.float32)
    b = np.array([0.4, 0.6], dtype=np.float32)
    out = combine_prob([a, b])
    assert np.allclose(out, [0.3, 0.7], atol=1e-6), out


def test_logit_matches_hand_calc():
    # two members both 0.5 -> logit 0 -> back to 0.5
    out = combine_logit([np.array([0.5]), np.array([0.5])])
    assert np.allclose(out, [0.5], atol=1e-6), out
    # 0.2 and 0.8 average in logit space -> symmetric -> 0.5 (not 0.5 in prob? it is here)
    out2 = combine_logit([np.array([0.2]), np.array([0.8])])
    assert np.allclose(out2, [0.5], atol=1e-6), out2


def test_logit_avoids_half_pileup():
    # prob-mean of an overconfident pair sits exactly at 0.5; logit-mean does not
    # pile at 0.5 for an asymmetric-confidence pair.
    a = np.array([0.99]); b = np.array([0.40])
    assert combine_logit([a, b])[0] > combine_prob([a, b])[0]


def test_rank_is_monotone_and_bounded():
    a = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    out = combine_rank([a])  # single member -> its own normalised ranks
    assert out.min() >= 0.0 and out.max() <= 1.0, out
    assert np.argmax(out) == 1 and np.argmin(out) == 0, out


def test_registry_keys():
    assert set(COMBINERS) == {"prob", "logit", "rank"}


def test_stacker_learns_to_favour_reliable_member():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, size=n)
    good = np.clip(y + rng.normal(0, 0.3, n), 0.01, 0.99)   # informative member
    noise = rng.uniform(0.01, 0.99, n)                       # useless member
    from combiners import fit_stacker, apply_stacker
    w, b = fit_stacker(np.stack([good, noise], axis=1), y, l2=1.0)
    assert abs(w[0]) > abs(w[1]), (w, b)                     # trusts the good member
    pred = apply_stacker(np.stack([good, noise], axis=1), w, b)
    acc = ((pred >= 0.5).astype(int) == y).mean()
    assert acc > 0.8, acc


def test_stacker_l2_shrinks_weights():
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, size=n)
    x = np.clip(y + rng.normal(0, 0.4, n), 0.01, 0.99)
    from combiners import fit_stacker
    w_lo, _ = fit_stacker(x[:, None], y, l2=0.1)
    w_hi, _ = fit_stacker(x[:, None], y, l2=100.0)
    assert abs(w_hi[0]) < abs(w_lo[0]), (w_lo, w_hi)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("ALL COMBINER TESTS PASSED")
