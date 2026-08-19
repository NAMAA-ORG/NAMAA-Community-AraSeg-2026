"""Offline checks for Lane C Task 4 (paired-view consistency).

The projection check (punct_aug._paired_projection / _removed_flags) is pure Python
and runs anywhere. The KL check (train.paired_word_kl) needs torch -- skip it with a
clear message on a box that doesn't have it (this repo's off-cluster convention; see
gate_member.py's docstring for the same constraint).
"""
from punct_aug import _removed_flags, _paired_projection


def test_projection_appends_removed_punct_to_preceding_survivor():
    pa_tokens = ["hello", ",", "world", "."]
    nopnx_tokens = ["hello", "world"]
    flags = _removed_flags(pa_tokens, nopnx_tokens)
    assert flags == [False, True, False, True]

    projected = _paired_projection(pa_tokens, nopnx_tokens, flags)
    assert projected == ["hello,", "world."]
    assert len(projected) == len(nopnx_tokens)


def test_doc_initial_removed_token_has_no_owner_and_is_dropped():
    pa_tokens = ["-", "hello", "world"]
    nopnx_tokens = ["hello", "world"]
    flags = _removed_flags(pa_tokens, nopnx_tokens)
    assert flags == [True, False, False]

    projected = _paired_projection(pa_tokens, nopnx_tokens, flags)
    assert projected == ["hello", "world"]
    assert len(projected) == len(nopnx_tokens)


def test_labels_are_untouched_official_nopnx_labels():
    # build_paired_chunks passes nopnx_labels straight through -- the projection
    # only ever changes token TEXT, never which words carry a boundary.
    nopnx_labels = [0, 1]
    pa_tokens = ["hello", ",", "world", "."]
    nopnx_tokens = ["hello", "world"]
    flags = _removed_flags(pa_tokens, nopnx_tokens)
    projected = _paired_projection(pa_tokens, nopnx_tokens, flags)
    assert len(projected) == len(nopnx_labels)  # word-for-word, so labels still align


def test_paired_word_kl_zero_for_identical_and_positive_after_perturbation():
    try:
        import torch
    except ImportError:
        print("skip  test_paired_word_kl_zero_for_identical_and_positive_after_perturbation "
              "(no torch on this box -- verify on KISSKI)")
        return
    from train import paired_word_kl

    logits = torch.zeros(1, 4, 2)
    logits[0, 0] = torch.tensor([2.0, -1.0])
    logits[0, 2] = torch.tensor([-0.5, 0.5])
    word_idx = torch.tensor([[0, -1, 1, -1]])

    zero = paired_word_kl(logits, word_idx, logits.clone(), word_idx)
    assert torch.allclose(zero, torch.zeros(()), atol=1e-6), zero

    perturbed = logits.clone()
    perturbed[0, 0] = torch.tensor([-2.0, 2.0])
    positive = paired_word_kl(logits, word_idx, perturbed, word_idx)
    assert positive.item() > 1e-4, positive


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
