import sys
from pathlib import Path
from typing import Any
from typing import Tuple
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from pysindy import AxesArray

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

reduction_methods = entry_points(group="ara.dim_reduction")

Pathlike = Union[str, Path]
STDataset = dict
DATA_DIR = Path("/home/ara/data").absolute()
st_loc = Path("01_matrix_HDF5")
NUMERICAL_PREFIX = "010"
DATAFILE_PREFIX = "simVectors" + NUMERICAL_PREFIX
vector_axes = {"ax_spatial": [0, 1], "ax_time": 2, "ax_coord": 3}


def _open(filename: Pathlike) -> STDataset:
    with h5py.File(filename) as h5file:
        return {
            "t1": np.array(h5file["t1"]),
            "t2": np.array(h5file["t2"]),
            "x": np.array(h5file["x"]),
            "y": np.array(h5file["y"]),
            "v1": AxesArray(
                np.stack((h5file["Vx1"], h5file["Vy1"], h5file["Vz1"]), axis=-1),
                axes=vector_axes,
            ),
            "v2": AxesArray(
                np.stack((h5file["Vx2"], h5file["Vy2"], h5file["Vz2"]), axis=-1),
                axes=vector_axes,
            ),
        }


def open(data_num: Union[int, str]) -> STDataset:
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


def svd_time(arr: AxesArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the SVD of arr, flattening all axes but time"""
    time_last = np.moveaxis(arr, arr.ax_time, -1)
    flattened = np.reshape(time_last, (-1, arr.n_time))
    svd = np.linalg.svd(flattened, full_matrices=False)
    return {"U": svd.U, "S": svd.S, "Vh": svd.Vh}


def save_dim_reduction(
    data_num: Union[int, str],
    reduction_type: str,
    suffix: str = "",
    **kwargs,
) -> Tuple[Any, Path]:
    """Create and cache a dimensionality reduction, applied to the data.

    Args:
        data_num: number that identifies the original data (e.g. the
            last three digits of the spatiotemporal h5 files
        reduction_type: the name of a reduction method registered to the
            ara.dim_reduction entry point.
        suffix: a custom suffix to append to the reducer name in
            creating the file name
        kwargs: arguments passed to reduction method, including the data

    Returns:
        The output of the reduction method and the path to the file.
    """
    try:
        reducer = reduction_methods[reduction_type].load()
    except KeyError:
        raise ValueError(
            f"No reduction method named {reduction_type} is installed."
            "Reduction methods need to be installed as an entry point to"
            "the 'ara.dim_reduction' group"
        )
    except AttributeError:
        raise AttributeError(f"Entry point {reduction_type} has no attribute 'names'")
    filename = f"{reducer.__name__}{suffix}{NUMERICAL_PREFIX}{data_num}.npz"
    filename = DATA_DIR / st_loc / filename
    if filename.exists():
        with np.load(filename, allow_pickle=False) as compressed:
            out = dict(compressed)
        return out, filename
    out = reducer(**kwargs)
    np.savez_compressed(filename, **out)
    return out, filename


def arr_to_movie(
    arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    filename: str,
    framestep: int = 10,
    **kwargs,
):
    """Create an animation of a simulation for v1 and v2

    Only animates the in-plane components (x and y) and not the
    orthogonal (z) components.

    Args:
        arr: Data array, shape is (nx * ny * 6, nt)
        x: x position of data
        y: y position of data
        filename: saved in current working directory
        framestep: number of timesteps in between each frame (default 10)
        kwargs: passed to matplotlib.Axes.quiver
    """
    nt = arr.shape[-1]
    nx = len(x)
    ny = len(y)
    vecs = np.reshape(arr, (nx, ny, 6, nt))
    vecs = np.moveaxis(vecs, -2, -1)
    Vx1 = vecs[..., 0]
    Vy1 = vecs[..., 1]
    Vx2 = vecs[..., 3]
    Vy2 = vecs[..., 4]

    # Create Animation Plot/File
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 6), sharex=True, sharey=True
    )
    plt.suptitle("Vectors, $it$ = " + str(0), fontsize=14)
    ax1 = axes[0]
    ax2 = axes[1]
    scale1 = 10 * np.max(np.abs(vecs[..., 0:2]))
    scale2 = 10 * np.max(np.abs(vecs[..., 3:5]))

    # To animate the quiver, we can change the u and v values, in animate() method.
    def animate(it):
        print(f"Making frame {it}", end="\r", flush=True)
        u1, v1 = Vx1[:, :, it], Vy1[:, :, it]
        u2, v2 = Vx2[:, :, it], Vy2[:, :, it]
        ax1.clear()
        ax2.clear()
        ln1 = ax1.quiver(x, y, u1, v1, color="b", scale=scale1, alpha=1, **kwargs)
        ln2 = ax2.quiver(x, y, u2, v2, color="g", scale=scale2, alpha=1, **kwargs)
        ax1.set_title("V1, $t$ = " + str(it))
        ax2.set_title("V2, $t$ = " + str(it))
        plt.suptitle("Vectors, $it$ = " + str(it), fontsize=14)
        return [ln1, ln2]

    # Create an animation object
    ani = animation.FuncAnimation(
        fig, animate, interval=framestep, frames=range(0, nt, framestep), repeat=False
    )

    # write animation as mp4
    writer = animation.FFMpegWriter(fps=30)
    ani.save(f"{filename}.mp4", writer=writer)
    return ani
