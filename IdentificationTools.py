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
        _assert_1D_float(num, "num")
        self.__num: FloatArray1D = num.astype(float, copy=False)
        self.__num_len: int = int(self.__num.shape[0])

        # 分母多項式
        _assert_1D_float(den, "den")
        self.__den: FloatArray1D = den.astype(float, copy=False)
        self.__den_len: int = int(self.__den.shape[0])

        # 遅れ時間
        assert delay >= 0, "delay must be non-negative"
        self.__delay: int = int(delay)

        # パラメータベクトルの構成（(num_len + den_len, 1)）
        self.__parameter: FloatArray2D = np.vstack(
            [self.__den.reshape(-1, 1), self.__num.reshape(-1, 1)]
        )

        # 入出力バッファ(初期値0)
        self.input_buf: FloatArray1D = np.zeros(self.__num_len + self.__delay, dtype=float)
        self.output_buf: FloatArray1D = np.zeros(self.__den_len, dtype=float)

        self.__regressor: FloatArray2D = np.zeros((self.__num_len + self.__den_len, 1), dtype=float)

    # 伝達関数の文字列表現
    def __str__(self) -> str:
        def poly_to_str(coeffs, start_power=1):
            terms = []
            for i, c in enumerate(coeffs, start=start_power):
                if abs(c) < 1e-12:
                    continue

                sign = "+" if c >= 0 else "-"
                coef = abs(c)

                if coef == 1.0:
                    term = f"q^-{i}"
                else:
                    term = f"{coef:.4g} q^-{i}"

                terms.append((sign, term))

            if not terms:
                return "0"

            # 先頭だけ符号を消す
            sgn, term = terms[0]
            expr = term if sgn == "+" else f"- {term}"

            for sgn, term in terms[1:]:
                expr += f" {sgn} {term}"

            return expr

        num_str = poly_to_str(self.__num)
        den_str = poly_to_str(self.__den)

        d = self.__delay

        lines = []
        lines.append("G(q) = y(k) / u(k)")
        lines.append(f"     = q^-{d} * ( {num_str} )")
        lines.append(f"       ---------------------")
        lines.append(f"         ( 1 {(' + ' + den_str) if den_str != '0' else ''} )")

        return "\n".join(lines)
    

    @property
    def den(self) -> FloatArray1D:
        return self.__den
    
    @property
    def num(self) -> FloatArray1D:
        return self.__num
    
    @property
    def parameter(self) -> FloatArray2D:
        return self.__parameter
    
    @property
    def regressor(self) -> FloatArray2D:
        return self.__regressor

    @parameter.setter
    def parameter(self, param: FloatArray2D) -> None:
        if (self.__parameter.shape != param.shape):
            raise ValueError(f"parameterの形状が不正です. parameter:{self.__parameter.shape}, param:{param.shape}")
        
        # パラメータベクトルの分割と代入
        self.__den[:] = param[0:self.__den_len].reshape(-1)
        self.__num[:] = param[self.__den_len:self.__den_len + self.__num_len].reshape(-1)

        self.__parameter = param

    def forward(self, u: Inputs) -> Outputs:
        """
        regressor: (num_len + den_len, 1)
        """
        # Inputs（タプルから最初の要素を取り出す）
        u = u[0]        

        if self.is_predict:
            self.updateInput(u)

        self.__regressor = np.vstack(
            [
                self.output_buf.reshape(-1, 1),
                self.input_buf[self.__delay:self.__delay+self.__num_len].reshape(-1, 1),
            ]
        )
        # y(k) = \theta^{\top} \phi(k)
        y:float = (self.__parameter.T @ self.__regressor).item()

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
        self.output_buf[0] = -float(y)

    # リグレッサーの状態リセット
    def reset(self) -> None:
        self.input_buf[:] = 0.0
        self.output_buf[:] = 0.0


class BJ(Function):
    
    def __init__(self, B: FloatArray1D, F: FloatArray1D, C: FloatArray1D, D: FloatArray1D, delay:int=0)->None:
        self.G = QtransferFunc(num=B, den=F, delay=delay, predict=True)
        self.H = QtransferFunc(num=C, den=D, delay=0, predict=True)

        self.__parameter: FloatArray2D = np.vstack([self.G.parameter, self.H.parameter])

    @property
    def parameter(self) -> FloatArray2D:
        return np.vstack([self.G.parameter, self.H.parameter])
    
    @parameter.setter
    def parameter(self, param: FloatArray2D) -> None:
        B_dim = self.G.num.shape[0]
        F_dim = self.G.den.shape[0]
        C_dim = self.H.num.shape[0]
        D_dim = self.H.den.shape[0]

        self.G.parameter = param[0:B_dim+F_dim]
        self.H.parameter = param[B_dim+F_dim:B_dim+F_dim+C_dim+D_dim]
    
    def forward(self, inputs: Inputs)-> Outputs:
        # 入力の取り出し
        u = inputs[0]
        e = inputs[1]

        return self.G(u) + self.H(e)
    

def PEM_residuals(param: np.ndarray, model_dim: Tuple[int,int,int,int], u: np.ndarray, y: np.ndarray) -> np.ndarray:
    B_dim, F_dim, C_dim, D_dim = model_dim
    

    # パラメータ分割
    B = param[0:B_dim]
    F = param[B_dim:B_dim+F_dim]
    G = QtransferFunc(num=B, den=F, delay=0, predict=True)

    C = param[B_dim+F_dim:B_dim+F_dim+C_dim]
    D = param[B_dim+F_dim+C_dim:B_dim+F_dim+C_dim+D_dim]
    H_inv = QtransferFunc(num=D, den=C, delay=0, predict=True)
    
    # データ整形
    u = np.asarray(u).reshape(-1)
    y = np.asarray(y).reshape(-1)
    N = y.shape[0]
    assert u.shape[0] == N

    # y_g と eps は 1サンプル後ろに入る（predict=True の仕様）
    y_g = np.zeros(N, dtype=float)
    eps = np.zeros(N, dtype=float)

    # 1) y_g(k) = G u
    for k in range(N):
        uk = np.array([[u[k]]], dtype=float)
        yg_kp1 = G((uk,))[0]
        if k + 1 < N:
            y_g[k+1] = float(yg_kp1)

    # 2) eps(k) = H^{-1} ( y(k) - y_g(k) )
    for k in range(N):
        wk = np.array([[y[k] - y_g[k]]], dtype=float)
        eps_kp1 = H_inv((wk,))[0]
        if k + 1 < N:
            eps[k+1] = float(eps_kp1)

    # 先頭はズレ＆初期条件がきついので捨てる（最低でも1個）
    return eps[1:]



        
if __name__ == "__main__":
    system = QtransferFunc(num=np.array([0.2]), den=np.array([-0.8]))

    u = np.zeros(100)
    u[10:-1] = 1
    output = np.zeros_like(u)
    test_output = np.zeros_like(u)
    
    for step in range(len(u)-1):
        output[step+1] = system(np.array([[u[step]]], dtype=float))[0]

        u_ = u[step] if step-1 >= 0 else 0.0
        test_output[step+1] = 0.2*u_ + 0.8*test_output[step]


    fig = plt.figure()
    ax_top = fig.add_subplot(2, 1, 1)
    ax_top.plot(output)
    ax_top.plot(test_output)
    ax_bottom = fig.add_subplot(2, 1, 2)
    ax_bottom.plot(u)
    plt.show()
