from src.core.dataset.split import dividir_dataset


def test_split_es_reproducible_y_disjunto():
    labels = [i for i in range(4) for _ in range(20)]
    split_a = dividir_dataset(labels, 0.70, 0.15, 0.15, 42)
    split_b = dividir_dataset(labels, 0.70, 0.15, 0.15, 42)

    assert split_a == split_b
    assert len(split_a.train) == 56
    assert len(split_a.val) == 12
    assert len(split_a.test) == 12
    assert not set(split_a.train) & set(split_a.val)
    assert not set(split_a.train) & set(split_a.test)
    assert not set(split_a.val) & set(split_a.test)
