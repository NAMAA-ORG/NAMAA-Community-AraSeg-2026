from gate_member import best_dev_subset


def test_best_dev_subset_ignores_test_score():
    systems = {
        ("e69",): {"dev_f1": 0.80, "test": {"f1": 0.99}},
        ("e71",): {"dev_f1": 0.82, "test": {"f1": 0.70}},
    }
    assert best_dev_subset(systems) == ("e71",)


def test_best_dev_subset_prefers_smaller_on_tie():
    systems = {
        ("e69",): {"dev_f1": 0.80, "test": {"f1": 0.50}},
        ("e69", "e71"): {"dev_f1": 0.80, "test": {"f1": 0.99}},
    }
    assert best_dev_subset(systems) == ("e69",)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
