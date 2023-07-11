import numpy as np

import ara


def test_open():
    dataset = ara.open(104)
    assert set(dataset.keys()) == {"t1", "t2", "v1", "v2", "x", "y"}
    assert dataset["v1"].shape == (119, 119, 1786, 3)
    assert dataset["t1"].shape == (1786,)


def test_list_datasets():
    datasets = ara.list_datasets()
    assert len(datasets) > 10


def test_time_misalignment():
    t1 = np.arange(0, 8, 2)
    t2 = np.arange(0.1, 8.1, 2)
    terr = ara.time_misalignment(t1, t2)
    assert np.abs(terr - 0.05) < 1e-15
