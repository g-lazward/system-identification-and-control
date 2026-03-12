import numpy as np
from typing import Any, Tuple
from numpy.typing import NDArray


FloatArray1D = NDArray[np.floating]
FloatArray2D = NDArray[np.floating]

# FloatArray2Dの型チェックを行う関数
def _assert_2D_float(variable: np.ndarray, name: str) -> None:
    if not isinstance(variable, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray")
    if variable.ndim != 2:
        raise ValueError(f"{name} must be 2D, ndim={variable.ndim}")
    if not np.issubdtype(variable.dtype, np.floating):
        raise TypeError(f"{name} must be floating dtype, dtype={variable.dtype}")

# FloatArray1Dの型チェックを行う関数
def _assert_1D_float(variable: np.ndarray, name: str) -> None:
    if not isinstance(variable, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray")
    if variable.ndim != 1:
        raise ValueError(f"{name} must be 1D, ndim={variable.ndim}")
    if not np.issubdtype(variable.dtype, np.floating):
        raise TypeError(f"{name} must be floating dtype, dtype={variable.dtype}")


Inputs = Tuple[FloatArray2D, ...]
Outputs = Tuple[FloatArray2D, ...]