import numpy as np
import pytest
from pysindy import AxesArray

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


@pytest.fixture
def fake_dataset():
    nt = 3
    nx = 2
    axes = {"ax_spatial": [0, 1], "ax_time": 2, "ax_coord": 3}
    return {
        "t1": np.arange(nt),
        "t2": np.arange(nt),
        "x": np.arange(nx),
        "y": np.arange(nx),
        "v1": AxesArray(np.ones((nx, nx, nt, 3)), axes),
        "v2": AxesArray(np.ones((nx, nx, nt, 3)), axes),
    }


@pytest.fixture
def saved_dim_red_data(fake_dataset):
    v = np.concatenate((fake_dataset["v1"], fake_dataset["v2"]), axis=-1)
    _, filename = ara.save_dim_reduction(104, "svd_time", "_test", arr=v)
    yield filename
    filename.unlink()


def test_dim_reduction_plugin(saved_dim_red_data):
    pass


def test_dim_reduction_cache(saved_dim_red_data, fake_dataset):
    v = np.concatenate((fake_dataset["v1"], fake_dataset["v2"]), axis=-1)
    expected = saved_dim_red_data.stat().st_mtime_ns
    _, filename2 = ara.save_dim_reduction(104, "svd_time", "_test", arr=v)
    result = filename2.stat().st_mtime_ns
    assert result == expected
