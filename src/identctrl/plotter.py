import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple

from identctrl.types import FloatArray1D, FloatArray2D, _assert_1D_float, _assert_2D_float

def pSpectrum(input: FloatArray1D, fs: float, 
              detrend: bool = True, window: str = "None")->Tuple[FloatArray1D, FloatArray1D]:
    """
    input: 1D signal
    fs: sampling frequency [Hz]
    detrend: remove mean (DC) before FFT
    window: None or "hann"
    Returns:
      f: frequency axis [Hz] (0..fs/2)
      P: power spectrum (one-sided) ~ |FFT|^2 (scaled)
    """
    # inputの型チェック
    _assert_1D_float(input, "input")
    
    if detrend:
        input = input - np.mean(input)
    
    N = len(input)
    
    if window == "hann":
        w = np.hanning(N)
        input = input * w
    
    # FFTを計算
    fft_result: np.ndarray = np.fft.fft(input)
    
    # 周波数軸を計算
    f: np.ndarray = np.fft.fftfreq(N, d=1/fs)
    
    # パワースペクトルを計算
    p: np.ndarray = (np.abs(fft_result)**2) / N
    
    return f, p