import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from identctrl.ident_types import FloatArray1D, FloatArray2D, _assert_1D_float, _assert_2D_float

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




def lowProjRegressor(regressor_matrix: np.ndarray, components: int, method: str = "svd") -> np.ndarray:
    """
    regressor_matrix: shape (n, T)
    method: "svd"
    """
    # 有効な時刻だけ取り出す
    valid_mask = ~np.isnan(regressor_matrix).any(axis=0)
    X = regressor_matrix[:, valid_mask].T

    match method:
        case "svd":
            # 平均を引く
            X_centered = X - np.mean(X, axis=0, keepdims=True)
            # SVD
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

            # 2次元へ射影
            Z = X_centered @ Vt[:components].T   # shape: (サンプル数, 2)

            return Z
        
import numpy as np
import matplotlib.pyplot as plt


def plotRegressor(Z: np.ndarray, lim: list = [-10, 10], animate: bool = False, show: bool = False, title: str = None)-> Union[plt.Figure, FuncAnimation]:
    """
    低次元に射影されたリグレッサをプロットする

    Parameters
    ----------
    Z : np.ndarray
        shape (T, d) の低次元データ (d=2 or 3)
    animate : bool
        時系列アニメーション表示
    """

    dim = Z.shape[1]

    if dim not in (2, 3):
        raise ValueError("Z must be 2D or 3D")

    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(title)

        if not animate:
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(len(Z)), cmap="viridis", s=10)
            plt.colorbar(sc, ax=ax, label="sample index")

        else:
            sc = ax.scatter([], [], s=10, c=[], cmap="viridis", vmin=0, vmax=len(Z)-1)
            def update(frame):
                pts = Z[:frame+1, :2]
                sc.set_offsets(pts)
                sc.set_array(np.arange(frame+1))
                return (sc,)

            animation = FuncAnimation(fig, update, frames=len(Z), interval=30)
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[0], lim[1])
        ax.set_xlabel("dim1")
        ax.set_ylabel("dim2")

    else:
        
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)

        if not animate:
            sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=np.arange(len(Z)), cmap="viridis", s=10)
            plt.colorbar(sc, ax=ax, label="sample index")

        else:
            
            sc = ax.scatter([], [], [], s=10)

            def update(frame):
                x = Z[:frame+1, 0]
                y = Z[:frame+1, 1]
                z = Z[:frame+1, 2]

                sc._offsets3d = (x, y, z)
                ax.view_init(elev=25, azim=frame * 0.8)
                return sc,

            animation = FuncAnimation(fig, update, frames=len(Z), interval=30)

        min, max = lim
        ax.set_xlim(min, max)
        ax.set_ylim(min, max)
        ax.set_zlim(min, max)

        ax.set_xlabel("dim1")
        ax.set_ylabel("dim2")
        ax.set_zlabel("dim3")
        
    if show:
      plt.show()
    return animation if animate else fig

def plotRegressorOnCircle(Z: np.ndarray, magnitude: np.ndarray | None = None, title: str = None):
    
    dim = Z.shape[1]

    if dim not in (2, 3):
        raise ValueError("Z must be 2D or 3D")

    if dim == 2:
        eps = 1e-12
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        Z_unit = Z / (norm + eps)

        if magnitude is None:
            magnitude = norm[:, 0]

        fig, ax = plt.subplots(figsize=(7, 7))

        # 単位円
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="dashed")

        sc = ax.scatter(
            Z_unit[:,0],
            Z_unit[:,1],
            c=magnitude,
            cmap="viridis",
            s=20
        )
        

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("magnitude")
    else:
        eps = 1e-12
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        Z_unit = Z / (norm + eps)

        if magnitude is None:
            magnitude = norm[:, 0]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)

        sc = ax.scatter(
            Z_unit[:, 0],
            Z_unit[:, 1],
            Z_unit[:, 2],
            c=magnitude,
            cmap="viridis",
            s=20
        )

        # 単位球を描く
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(0, np.pi, 40)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(
            x, y, z,
            color="gray",
            alpha=0.1,
            linewidth=0.2,
            zorder=0
        )

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_box_aspect([1,1,1])
    return fig