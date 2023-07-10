from pathlib import Path

import numpy as np

import h5py

Pathlike = str | Path
STDataset = dict

def _open(filename: Pathlike) -> STDataset:
    with h5py.File(filename) as h5file:
        return {
            "t1": h5file["t1"],
            "t2": h5file["t2"],
            "x": h5file["x"],
            "y": h5file["y"],
            "v1": np.stack((h5file["Vx1"], h5file["Vy1"], h5file["Vz1"]), axis=-1),
            "v2": np.stack((h5file["Vx2"], h5file["Vy2"], h5file["Vz2"]), axis=-1)
        }
    

def open(data_num: int | str) -> STDataset:
    """Open a dataset in the default data directory on doppio

    Args:
        data_num: the trial id, e.g. 104.  "simVectors010" is prepended
            to create the filename

    Returns:
        dictionary with keys to the spatiotemporal grid and vectors
    """
    DATA_DIR = Path("/home/ara/data")
    st_loc = Path("01_matrix_HDF5")
    DATAFILE_PREFIX = "simVectors010"
    filename = DATAFILE_PREFIX + str(data_num) + ".h5"
    return _open(DATA_DIR / st_loc / filename)
