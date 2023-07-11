from pathlib import Path

import h5py
import numpy as np

Pathlike = str | Path
STDataset = dict
DATA_DIR = Path("/home/ara/data")
st_loc = Path("01_matrix_HDF5")
DATAFILE_PREFIX = "simVectors010"


def _open(filename: Pathlike) -> STDataset:
    with h5py.File(filename) as h5file:
        return {
            "t1": np.array(h5file["t1"]),
            "t2": np.array(h5file["t2"]),
            "x": np.array(h5file["x"]),
            "y": np.array(h5file["y"]),
            "v1": np.stack((h5file["Vx1"], h5file["Vy1"], h5file["Vz1"]), axis=-1),
            "v2": np.stack((h5file["Vx2"], h5file["Vy2"], h5file["Vz2"]), axis=-1),
        }


def open(data_num: int | str) -> STDataset:
    """Open a dataset in the default data directory on doppio

    Args:
        data_num: the trial id, e.g. 104.  "simVectors010" is prepended
            to create the filename

    Returns:
        dictionary with keys to the spatiotemporal grid and vectors
    """
    filename = DATAFILE_PREFIX + str(data_num) + ".h5"
    return _open(DATA_DIR / st_loc / filename)


def _list_datasets():
    matches = Path(DATA_DIR / st_loc).glob(DATAFILE_PREFIX + "*.h5")
    return tuple(matches)


def list_datasets():
    """List available datasets in the default data directory on doppio

    Returns:
        tuple of datasets, by number, that can be passed to ara.open()
    """
    matches = _list_datasets()
    return tuple(match.name[13:-3] for match in matches)


def time_misalignment(t1: np.ndarray, t2: np.ndarray) -> float:
    """Calculate how misaligned the times of v1 and v2 are.

    Args:
        t1, t2: the times of v1 and v2 observations

    Returns:
        Maximum value of time delta between equivaluent t1 and t2
        indices, divided by the t1 or t2 gap on either side of the index
    """
    dts1 = t1[1:] - t1[:-1]
    dts2 = t2[1:] - t2[:-1]
    terr_rel = np.stack(
        (
            (t2 - t1)[1:] / dts1,
            (t2 - t1)[:-1] / dts1,
            (t2 - t1)[1:] / dts2,
            (t2 - t1)[:-1] / dts2,
        )
    )
    return np.max(np.abs(terr_rel))
