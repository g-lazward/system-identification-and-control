from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple

# main関数内のプロットに使用
import matplotlib.pyplot as plt

FloatArray1D = NDArray[np.floating]
FloatArray2D = NDArray[np.floating]
Inputs = Tuple[FloatArray2D, ...]
Outputs = Tuple[FloatArray2D, ...]

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


class Function(ABC):
    def __call__(self, *inputs: FloatArray2D, **params):
        # 信号類はinputs(タプル)，パラメータ等はparamsに渡して処理

        # 型チェック
        if len(inputs) == 0:
            raise ValueError("inputs must have at least one element")
        for i, input in enumerate(inputs):
            _assert_2D_float(input, f"input[{i}]")

        return self.forward(inputs, **params)

    @abstractmethod
    def forward(self, *input: FloatArray2D, **params) -> Outputs:
        raise NotImplementedError


class QtransferFunc(Function):
    # num, denは1次元配列として渡すこと
    def __init__(self, num: FloatArray1D, den: FloatArray1D, delay: int = 0, predict:bool=True) -> None:
        # Trueの場合, y(k+1) = \theta^{\top}\phi(k)
        # Falseの場合, y(k) = \theta^{\top}\phi(k) *phiの中身は若干異なる．
        self.is_predict = predict

        # 分子多項式
        assert num.ndim == 1, "numの次元が不正です"
        self.num: FloatArray1D = num.astype(float, copy=False)
        self.num_len: int = int(self.num.shape[0])

        # 分母多項式
        assert den.ndim == 1, "denの次元が不正です"
        self.den: FloatArray1D = den.astype(float, copy=False)
        self.den_len: int = int(self.den.shape[0])

        # 遅れ時間
        self.delay: int = int(delay)

        # パラメータベクトルの構成（(num_len + den_len, 1)）
        self.parameter: FloatArray2D = np.vstack(
            [self.num.reshape(-1, 1), -self.den.reshape(-1, 1)]
        )

        # 入出力バッファ(初期値0)
        self.input_buf: FloatArray1D = np.zeros(self.num_len + self.delay, dtype=float)
        self.output_buf: FloatArray1D = np.zeros(self.den_len, dtype=float)

    def forward(self, u: Inputs) -> Outputs:
        """
        regressor: (num_len + den_len, 1)
        """
        # Inputs（タプルから最初の要素を取り出す）
        u = u[0]        

        if self.is_predict:
            self.updateInput(u)

        regressor: FloatArray2D = np.vstack(
            [
                self.input_buf[self.delay:self.delay+self.num_len].reshape(-1, 1),
                self.output_buf.reshape(-1, 1),
            ]
        )
        # y(k) = \theta^{\top} \phi(k)
        y:float = (self.parameter.T @ regressor).item()

        self.updateOutput(y)

        if not self.is_predict:
            self.updateInput(u)

        return (y, )

    # リグレッサの入力ベクトルの更新
    def updateInput(self, u: FloatArray2D) -> None:
        self.input_buf[1:] = self.input_buf[:-1]
        self.input_buf[0] = float(u.item())
    
    # リグレッサの出力ベクトルの更新
    def updateOutput(self, y: float) -> None:
        self.output_buf[1:] = self.output_buf[:-1]
        self.output_buf[0] = float(y)


class BJmodel(Function):
    
    def __init__(self, B, F, C, D, delay=0):
        self.G = QtransferFunc(num=B, den=F, delay=delay, predict=True)
        self.H = QtransferFunc(num=C, den=D, delay=0, predict=True)
    
    def forward(self, u, e):
        return self.G(u) + self.G(e)





        
if __name__ == "__main__":
    system = QtransferFunc(num=np.array([0.2]), den=np.array([-0.8]))

    u = np.zeros(100)
    u[10:-1] = 1
    output = np.zeros_like(u)
    test_output = np.zeros_like(u)
    
    for step in range(len(u)-1):
        output[step+1] = system.forward(np.array([[u[step]]], dtype=float))

        u_ = u[step] if step-1 >= 0 else 0.0
        test_output[step+1] = 0.2*u_ + 0.8*test_output[step]


    fig = plt.figure()
    ax_top = fig.add_subplot(2, 1, 1)
    ax_top.plot(output)
    ax_top.plot(test_output)
    ax_bottom = fig.add_subplot(2, 1, 2)
    ax_bottom.plot(u)
    plt.show()
