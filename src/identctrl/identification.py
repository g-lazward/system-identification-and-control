from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from identctrl.types import FloatArray1D, FloatArray2D, _assert_1D_float, _assert_2D_float, Inputs, Outputs

# main関数内のプロットに使用
import matplotlib.pyplot as plt


class Function(ABC):
    """
    信号処理・システムモデル用の抽象基底クラス。

    このクラスは「関数オブジェクト」として振る舞うことを想定しており，
    インスタンスをそのまま callable にして使用できる。

    入力信号はすべて 2 次元の浮動小数点配列として受け取り，
    共通の型チェックを行った上で，実際の計算処理は `forward` メソッドに委譲する。

    具体的な処理内容はサブクラス側で `forward` を実装することで定義する。
    """

    def __call__(self, *inputs: FloatArray2D, **params):
        """
        入力信号とパラメータを受け取り，forward 処理を実行する。

        このメソッドにより，Function クラスのインスタンスは
        通常の関数のように呼び出すことができる。

        入力は可変長引数として受け取り，すべての入力信号に対して
        2 次元配列かどうかの型チェックを行う。
        パラメータ類はキーワード引数として受け取る。

        Args:
            *inputs (FloatArray2D): 入力信号を表す 2 次元の浮動小数点配列．
            **params: モデル係数やオプションなどの追加パラメータ．

        Returns:
            Outputs: forward メソッドによって計算された出力．

        Raises:
            ValueError: 入力信号が 1 つも与えられなかった場合．
            AssertionError: 入力信号が 2 次元の浮動小数点配列でない場合．
        """
        
        # 型チェック
        if len(inputs) == 0:
            raise ValueError("inputs must have at least one element")
        for i, input in enumerate(inputs):
            _assert_2D_float(input, f"input[{i}]")

        return self.forward(inputs, **params)

    @abstractmethod
    def forward(self, *input: FloatArray2D, **params) -> Outputs:
        """
        実際の信号処理・モデル計算を定義する抽象メソッド。

        このメソッドはサブクラスで必ず実装する必要があり，
        入力信号から出力をどのように計算するかを記述する。

        通常は 1 ステップ予測，シミュレーション，
        あるいはシステムの出力計算などをここで行う。

        Args:
            *input (FloatArray2D): 入力信号を表す 2 次元の浮動小数点配列．
            **params: 計算に必要なパラメータや設定値．

        Returns:
            Outputs: 計算結果としての出力信号．

        Raises:
            NotImplementedError: サブクラスで実装されていない場合．
        """
        raise NotImplementedError


class QtransferFunc(Function):
    """
    離散時間SISOの伝達関数モデル（q^-1 表現）を扱うクラス。

    分子 `num` と分母 `den`（どちらも 1 次元配列）と遅れ `delay` を持ち，
    内部バッファからリグレッサ φ を作って y = θ^T φ を計算する。

    `predict=True` のときは 1-step ahead 予測の並び（入力の更新タイミングが先）で動作する。
    """
    
    def __init__(self, num: FloatArray1D, den: FloatArray1D, delay: int = 0, predict:bool=True) -> None:
        """
        モデル係数と内部状態（バッファ等）を初期化する。

        Args:
            num (FloatArray1D): 分子多項式係数（q^-1 の係数列）．
            den (FloatArray1D): 分母多項式係数（q^-1 の係数列）．
            delay (int): 入力の遅れステップ数（0以上）．
            predict (bool): 予測モードの切替．Trueで 1-step ahead 系の更新順になる．

        Raises:
            AssertionError: delay が負の場合．
            AssertionError: num/den が 1 次元の浮動小数点配列でない場合．
        """

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

    def __str__(self) -> str:
        """
        伝達関数 G(q) をテキストとして整形して返す。

        Returns:
            str: q^-d と多項式を用いた G(q)=y/u の表示文字列．
        """

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
    def regressor(self) -> FloatArray2D:
        return self.__regressor
    
    @regressor.setter
    def regressor(self, phi: FloatArray2D) -> None:
        if self.__regressor.shape != phi.shape:
            raise ValueError(f"regressorの形状が不正です. regressor:{self.__regressor.shape}, phi:{phi.shape}")
        self.__regressor = phi
    
    @property
    def parameter(self) -> FloatArray2D:
        """
        パラメータベクトル θ を返す（[den; num] の縦結合）。

        Returns:
            FloatArray2D: 形状 (den_len + num_len, 1) のパラメータベクトル．
        """

        return self.__parameter
    
    @parameter.setter
    def parameter(self, param: FloatArray2D) -> None:
        """
        パラメータベクトル θ を更新し，den/num に反映する。

        Args:
            param (FloatArray2D): 形状が一致する新しいパラメータベクトル．

        Raises:
            ValueError: 形状が一致しない場合．
        """

        if (self.__parameter.shape != param.shape):
            raise ValueError(f"parameterの形状が不正です. parameter:{self.__parameter.shape}, param:{param.shape}")
        
        # パラメータベクトルの分割と代入
        self.__den[:] = param[0:self.__den_len].reshape(-1)
        self.__num[:] = param[self.__den_len:self.__den_len + self.__num_len].reshape(-1)

        self.__parameter = param

    def forward(self, u: Inputs) -> Outputs:
        """
        入力からリグレッサ φ を構成し，y = θ^T φ を計算して返す。

        predict の設定に応じて入力バッファの更新タイミングが変わる。

        Args:
            u (Inputs): 先頭要素に入力信号（2次元配列）が入ったタプル．

        Returns:
            Outputs: 出力 y を 1 要素タプルで返す．
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

    
    def updateInput(self, u: FloatArray2D) -> None:
        """
        入力バッファをシフトして最新入力を格納する。

        Args:
            u (FloatArray2D): 最新入力（スカラー相当の2次元配列）．
        """
        self.input_buf[1:] = self.input_buf[:-1]
        self.input_buf[0] = float(u.item())
    
    def updateOutput(self, y: float) -> None:
        """
        出力バッファをシフトして最新出力を格納する。
        （リグレッサ用に符号反転して保存する）

        Args:
            y (float): 最新出力．
        """
        self.output_buf[1:] = self.output_buf[:-1]
        self.output_buf[0] = -float(y)

    def reset(self) -> None:
        """
        入出力バッファをゼロに戻し，内部状態をリセットする。
        """
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

class RLS(Function):
    """Recursive Least Squares (RLS) estimator.

    This class implements the standard RLS parameter update rule:

        theta(t) = theta(t-1)
                   + P(t-1) phi(t)
                     (y(t) - phi(t)^T theta(t-1))
                     / (1 + phi(t)^T P(t-1) phi(t))

        P(t-1) = P(t-2)
               - P(t-2) phi(t-1) phi(t-1)^T P(t-2)
                 / (1 + phi(t-1)^T P(t-2) phi(t-1))

    Attributes:
        theta (FloatArray2D): Current parameter estimate vector.
        P (FloatArray2D): Current covariance matrix.
    """

    def __init__(self, theta_init: FloatArray2D, P_init: FloatArray2D, discount: float = 1.0) -> None:
        """Initializes the RLS estimator.

        Args:
            theta_init (FloatArray2D): Initial parameter vector.
                Shape should be (n, 1).
            P_init (FloatArray2D): Initial covariance matrix.
                Shape should be (n, n).

        Raises:
            AssertionError: If inputs are not 2D float arrays.
        """
        _assert_2D_float(theta_init, "theta_init")
        self.theta: FloatArray2D = theta_init
        _assert_2D_float(P_init, "P_init")
        self.P: FloatArray2D = P_init

        assert discount > 0 and discount <= 1, f"discount factor must be in (0, 1]; discount={discount}"
        self.discount: float = discount
    
    def forward(self, inputs: Inputs) -> Outputs:
        """Performs one RLS update step.

        Args:
            inputs (Inputs): Tuple containing:
                - phi (FloatArray2D): Regressor vector of shape (n, 1).
                - y (float): Measured output scalar.

        Returns:
            Outputs: Tuple containing updated parameter vector (theta,).
        """        
        phi = inputs[0]
        y = inputs[1]
        
        self.__calcTheta(phi, y)
        self.__calcP(phi)

        return (self.theta, )

    def __calcTheta(self, phi: FloatArray2D, y: float) -> None:
        """Updates the parameter vector theta.

        Args:
            phi (FloatArray2D): Regressor vector of shape (n, 1).
            y (float): Measured output scalar.
        """        
        self.theta = self.theta + self.P @ phi * (y - phi.T @ self.theta) / (self.discount + phi.T @ self.P @ phi)
    
    def __calcP(self, phi: FloatArray2D) -> None:
        """Updates the covariance matrix P.

        Args:
            phi (FloatArray2D): Regressor vector of shape (n, 1).
        """        
        self.P = (self.P - self.P @ phi @ phi.T @ self.P / (self.discount + phi.T @ self.P @ phi)) / self.discount

    def forward(self, inputs: Inputs) -> Outputs:
        """Performs one RLS update step.

        Args:
            inputs (Inputs): Tuple containing:
                - phi (FloatArray2D): Regressor vector of shape (n, 1).
                - y (float): Measured output scalar.

        Returns:
            Outputs: Tuple containing updated parameter vector (theta,).
        """        
        phi = inputs[0]
        y = inputs[1]
        
        self.__calcTheta(phi, y)
        self.__calcP(phi)

        return (self.theta, )



class MFRLS(RLS):
    """
    Bruce, Adam L., Ankit Goel, and Dennis S. Bernstein. 
    "Recursive least squares with matrix forgetting."
    2020 American Control Conference (ACC). IEEE, 2020.

    """
    def __init__(self, theta_init: FloatArray2D, P_init: FloatArray2D, discount: float) -> None:
        super().__init__(theta_init, P_init, discount)    
    
    def __calcTheta(self, phi: FloatArray2D, y: float) -> None:
        """Updates the parameter vector theta.

        Args:
            phi (FloatArray2D): Regressor vector of shape (n, 1).
            y (float): Measured output scalar.
        """        
        self.theta = self.theta + self.P @ phi * (y - phi.T @ self.theta)
    
    def __calcP(self, phi: FloatArray2D) -> None:
        """Updates the covariance matrix P.

        Args:
            phi (FloatArray2D): Regressor vector of shape (n, 1).
        """
        sigma, U = np.linalg.eigh(self.P)
        psi = (phi.T @ U).reshape(-1)
        # ノイズレベルに応じた閾値を設定
        esp = 1e-2
        
        diag = np.zeros_like(psi)
        for i in range(len(psi)):
            diag[i] = np.sqrt(self.discount) if np.linalg.norm(psi[i]) > esp else 1.0
        Lambda = np.diag(1/diag)

        B = U @ Lambda @ U.T
        L = B @ self.P @ B.T
        self.P = L - (L @ phi @ phi.T @ L)/ (1.0 + phi.T @ L @ phi)
        
    def forward(self, inputs: Inputs) -> Outputs:
        """Performs one RLS update step.

        Args:
            inputs (Inputs): Tuple containing:
                - phi (FloatArray2D): Regressor vector of shape (n, 1).
                - y (float): Measured output scalar.

        Returns:
            Outputs: Tuple containing updated parameter vector (theta,).
        """        
        phi = inputs[0]
        y = inputs[1]
        
        self.__calcP(phi)
        self.__calcTheta(phi, y)

        return (self.theta, )

class DFRLS(RLS):
    """
    Cao, Liyu, and Howard Schwartz. 
    "A directional forgetting algorithm based on the decomposition of the information matrix." 
    Automatica 36.11 (2000): 1725-1731.
    """
    def __init__(self, theta_init: FloatArray2D, P_init: FloatArray2D, epsilon: float, discount: float = 1.0) -> None:
        super().__init__(theta_init, P_init, discount)
        assert epsilon > 0, f"epsilon must be positive; epsilon={epsilon}"
        self.epsilon: float = epsilon

        self.R = np.linalg.inv(P_init)
    
    def __calcF(self, phi: FloatArray2D) -> np.ndarray:
        if np.linalg.norm(phi) > self.epsilon:
            M = (1-self.discount) * (self.R @ phi @ phi.T)/(phi.T @ self.R @ phi)
        else:
            M = np.zeros_like(self.R)
        return np.eye(self.R.shape[0]) - M
    
    def __calcR(self, phi: FloatArray2D) -> None:
        F = self.__calcF(phi)
        self.R = F @ self.R + phi @ phi.T
    
    def __calcP(self, phi: FloatArray2D) -> None:
        if np.linalg.norm(phi) > self.epsilon:
            modified_P = self.P + (1-self.discount)/self.discount * (phi @ phi.T)/(phi.T @ self.R @ phi)
        else:
            modified_P = self.P
        
        self.P = modified_P - (modified_P @ phi @ phi.T @ modified_P) / (1 + phi.T @ modified_P @ phi)
    
    def __calcTheta(self, phi: FloatArray2D, y: float) -> None:
        self.theta = self.theta + self.P @ phi * (y - phi.T @ self.theta)
            
        
    def forward(self, inputs):
        phi = inputs[0]
        y = inputs[1]
        
        self.__calcP(phi)
        self.__calcTheta(phi, y)
        self.__calcR(phi)
        

        return (self.theta, )






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
