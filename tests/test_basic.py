import ara

def test_open():
    dataset = ara.open(104)
    assert set(dataset.keys()) == {"t1", "t2", "v1", "v2", "x", "y"}
    assert dataset["v1"].shape == (119, 119, 1786, 3)


def test_list_datasets():
    datasets = ara.list_datasets()
    assert len(datasets) > 10