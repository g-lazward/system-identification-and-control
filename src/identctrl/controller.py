import numpy as np
from numpy.typing import NDArray
from typing import Union
from dataclasses import dataclass
from identctrl import identification as ident
from identctrl.ident_types import FloatArray1D, FloatArray2D

from cvxopt import matrix
from cvxopt.solvers import qp


@dataclass
class LDss:
    """
    線形離散時間状態空間モデルを表すデータクラス。

    次の状態方程式・出力方程式で表される
    線形離散時間システムを扱う。

        x(k+1) = A x(k) + B u(k)
        y(k)   = C x(k)

    本クラスはモデルパラメータを保持するための
    軽量なコンテナとして使用され，
    MPCや状態推定器などの制御アルゴリズムから参照される。

    Attributes
    ----------
    A : ndarray
        状態遷移行列
    B : ndarray
        入力行列
    C : ndarray
        出力行列
    """

    A: FloatArray2D
    B: FloatArray2D
    C: FloatArray2D

class MPC(ident.Function):
    """
    モデル予測制御（Model Predictive Control; MPC）クラス。

    線形離散時間状態空間モデルに基づき，
    予測ホライズン Hp，制御ホライズン Hu を用いた
    有限時間最適化問題を逐次解くことで制御入力を生成する。

    内部では，
    - 予測行列 Psi
    - 入力影響行列 Upsilon
    - 畳み込み行列 Theta
    を構成し，二次計画問題（QP）として入力差分を計算する。

    Luenberger 観測器を用いて状態推定も同時に行う。
    """
    def __init__(self, system: LDss, 
                 Hp: int, # 予測ホライズン
                 Hu: int, # 制御ホライズン
                 Hw: int, # 窓パラメータ
                 Q: FloatArray2D,  # 重み行列（誤差）
                 R: FloatArray2D,  # 重み行列（入力）
                 L: FloatArray2D,  # Luenberger観測器の重み
                 init_state: FloatArray2D = None, # 初期状態ベクトル
                 init_input: FloatArray2D = None # 初期入力
                 ) -> None:
        """
        MPCコントローラを初期化する。

        制御対象モデル，ホライズン長，重み行列，
        および初期状態・初期入力を設定し，
        MPCで使用する各種行列を事前計算する。

        Parameters
        ----------
        system : LDss
            離散時間状態空間モデル
        Hp : int
            予測ホライズン
        Hu : int
            制御ホライズン（Hp <= Hu を満たす必要がある）
        Hw : int
            窓パラメータ
        Q : ndarray
            出力誤差に対する重み行列（Hp × Hp）
        R : ndarray
            入力差分に対する重み行列（Hu × Hu）
        L : ndarray
            Luenberger観測器のゲイン行列
        init_state : ndarray, optional
            初期状態ベクトル (n, 1)
        init_input : ndarray, optional
            初期入力ベクトル (m, 1)

        Raises
        ------
        AssertionError
            行列サイズやホライズン条件が満たされていない場合
        """

        self.system: LDss = system

        # システムの次元
        self.system_dim:int = self.system.A.shape[0]
        # 入力のベクトルサイズ
        self.input_dim:int = self.system.B.shape[1]
        # 出力のベクトルサイズ
        self.output_dim:int = self.system.C.shape[0]        
        
        self.Hp = int(Hp)
        self.Hu = int(Hu)
        assert self.Hp >= self.Hu, "Hp >= Huを満たしていません"

        self.Hw = int(Hw)

        # 誤差に対する重み行列(Qが準正定かどうかの検証は行わない)
        self.Q = Q
        assert self.Q.shape == (Hp, Hp), f"Qのサイズは({Hp}, {Hp})が正しいです．現在:{self.Q.shape}"
        # 入力の差分に対する重み行列(Rが準正定かどうかの検証は行わない)
        self.R = R
        assert self.R.shape == (Hu, Hu), f"Rのサイズは({Hu}, {Hu})が正しいです．現在:{self.R.shape}"
        # Luenberger観測器の重み
        self.L = L
        assert self.L.shape == (self.system_dim, self.output_dim), f"Lのサイズは({self.system_dim}, {self.output_dim})が正しいです．現在:{self.L.shape}"

        # 初期状態ベクトル
        if init_state is None:
            self.state: FloatArray2D = np.zeros((self.system_dim, 1))
        else:
            assert init_state.shape == (self.system_dim, 1), "状態ベクトルは(n, 1)で与えてください"
            self.state: FloatArray2D = init_state
        
        # 初期入力
        if init_input is None:
            self.u = np.zeros((self.input_dim, 1))
        else:
            assert init_input.shape == (self.input_dim, 1), f"入力ベクトルの次元は({self.input_dim}, 1)で与えてください(Bの列数)"
            self.u = init_input

        self.Psi: FloatArray2D = self._calcPsi()
        self.Upsilon: FloatArray2D = self._calcUpsilon()
        self.Theta: FloatArray2D = self._calcTheta(self.Upsilon)

        self.P:FloatArray2D = 2*(self.Theta.T @ self.Q @ self.Theta + self.R)


    def _calcPsi(self)->FloatArray2D:
        """
        予測行列 Psi を計算する。

        Psi は現在の状態から将来の出力を予測するための行列であり，
        システム行列 A, C を用いて
        [C A; C A^2; ...; C A^Hp] の形で構成される。

        Returns
        -------
        ndarray
            予測行列 Psi (output_dim*Hp, system_dim)
        """        
        n = self.system_dim
        m = self.output_dim
        Hp = self.Hp
        A = self.system.A
        C = self.system.C

        Psi = np.zeros((m*Hp, n))
        A_power = A.copy()
        for i in range(1, Hp+1):
            Psi[(i-1)*m:i*m, :] = C@A_power
            A_power = A_power @ A
        return Psi

    def _calcUpsilon(self)->FloatArray2D:
        """
        入力影響行列 Upsilon を計算する。

        Upsilon は過去および現在の入力が
        将来の出力に与える影響を表す行列であり，
        C A^k B の累積和として構成される。

        Returns
        -------
        ndarray
            入力影響行列 Upsilon (output_dim*Hp, input_dim)
        """        
        A = self.system.A
        B = self.system.B
        C = self.system.C
        Hp = self.Hp
        m = self.output_dim
        l = self.input_dim

        Upsilon = np.zeros((m*Hp, l))
        A_power = A.copy()
        Upsilon[0:m, :] = C @ B
        for i in range(2, Hp+1):
            Upsilon[(i-1)*m:i*m][:] = Upsilon[(i-2)*m:(i-1)*m][:] + C @ A_power @ B
            A_power = A_power @ A
        return Upsilon

    def _calcTheta(self, Upsilon)->FloatArray2D:
        """
        畳み込み行列 Theta を計算する。

        Theta は制御ホライズン Hu に渡る入力差分系列と
        将来出力との関係を表す行列であり，
        Upsilon をブロック構造で配置することで構成される。

        Parameters
        ----------
        Upsilon : ndarray
            入力影響行列

        Returns
        -------
        ndarray
            畳み込み行列 Theta (output_dim*Hp, input_dim*Hu)
        """

        Hp = self.Hp
        Hu = self.Hu
        m = self.output_dim
        l = self.input_dim

        Theta = np.zeros((m*Hp, l*Hu))
        Theta[:, 0:l] = Upsilon
        for i in range(1, Hu):
            Theta[m*i:, i*l:(i+1)*l] = Upsilon[:m*(Hp-i), :]
        return Theta

    def _calcInput(self, state: FloatArray2D, prev_u: FloatArray2D, T: FloatArray2D):
        """
        最適な入力差分を二次計画問題として計算する。

        評価関数
            J           
        を最小化するQPを解き，
        最初のステップの入力差分のみを返す。

        Parameters
        ----------
        state : ndarray
            現在の状態ベクトル
        prev_u : ndarray
            直前の入力ベクトル
        T : ndarray
            参照信号ベクトル

        Returns
        -------
        ndarray
            次時刻に適用する入力差分ベクトル
        """        
        P = matrix(self.P)
        q = matrix(2 * self.Theta.T @ self.Q @ (self.Psi @ state + self.Upsilon @ prev_u - T))

        solution = qp(P=P, q=q)
        delta_U = np.array(solution["x"]).reshape(self.input_dim*self.Hu, 1)

        return delta_U[:self.input_dim, :]

    def _observeState(self, state, u, y):
        """
        Luenberger観測器による状態推定を行う。

        観測器の更新式
            x(k+1) = (A - L C)x(k) + B u(k) + L y(k)
        に基づいて次状態を推定する。

        Parameters
        ----------
        state : ndarray
            推定状態ベクトル
        u : ndarray
            入力ベクトル
        y : ndarray
            出力ベクトル

        Returns
        -------
        ndarray
            次時刻の推定状態ベクトル

        Raises
        ------
        AssertionError
            入力ベクトルの次元が不正な場合
        """

        assert state.shape == (self.system_dim, 1)
        assert u.shape == (self.input_dim, 1)
        assert y.shape == (self.output_dim, 1)

        next_state = (self.system.A - self.L @ self.system.C) @ state + self.system.B @ u + self.L @ y
        return next_state

    def forward(self, inputs: ident.Inputs)->ident.Outputs:
        """
        MPCのメイン処理を実行する。

        参照信号と現在の出力を受け取り，
        最適制御入力 u(k|k) と
        次時刻の推定状態 x̂(k+1|k) を計算する。

        Parameters
        ----------
        inputs : tuple
            inputs[0] : ndarray
                参照信号ベクトル
            inputs[1] : ndarray
                制御対象の出力ベクトル

        Returns
        -------
        list
            [制御入力ベクトル, 次時刻の推定状態ベクトル]

        Raises
        ------
        AssertionError
            参照信号ベクトルのサイズが不正な場合
        """

        T: FloatArray2D = inputs[0]
        y: FloatArray2D = inputs[1]
        
        assert T.shape == (self.output_dim*self.Hp, 1), f"Tは(output_dim*Hp, 1)のサイズで与えてください．現在：output_dim:{self.output_dim}, Hp:{self.Hp}"
        
        # 入力の差分を計算
        du = self._calcInput(self.state, self.u, T)
        self.u = self.u + du

        # 次のステップの推定状態ベクトルの計算(時間ステップに注意)
        # \hat{x}(k+1|k) = observer(\hat{x}(k|k-1), u(k), y(k))
        self.state = self._observeState(self.state, self.u, y)
        
        return [self.u, self.state]



class SimplePIDController(ident.Function):
    """
    単純なPID制御器クラス。

    比例（P），積分（I），微分（D）制御を組み合わせた
    離散時間PIDコントローラを実装する。

    誤差の積分値と1ステップ前の誤差を内部状態として保持し，
    `forward` が呼ばれるたびに制御入力を計算する。
    """
    def __init__(self, kp:float, ki:float, kd:float, Ts:float=1.0)->None:
        """
        PID制御器を初期化する。

        比例・積分・微分ゲインとサンプリング周期を設定し，
        内部状態（積分誤差，直前の誤差）を初期化する。

        Parameters
        ----------
        kp : float
            比例ゲイン
        ki : float
            積分ゲイン
        kd : float
            微分ゲイン
        Ts : float, optional
            サンプリング周期, by default 1.0
        """        
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.Ts = float(Ts)

        self.e_sum: float = 0
        self.prev_e: float = 0

    def forward(self, u: ident.Inputs)->ident.Outputs:
        """
        PID制御則に基づいて制御入力を計算する。

        入力として誤差信号を受け取り，
        比例項・積分項・微分項を用いて制御量を算出する。

        積分誤差および直前の誤差は内部状態として更新される。

        Parameters
        ----------
        u : Inputs
            u[0] に誤差信号（スカラー）を含む入力タプル

        Returns
        -------
        Outputs
            PID制御によって計算された制御入力（タプル）

        Notes
        -----
        本実装ではアンチワインドアップ処理や
        出力飽和処理は行っていない。
        """        
        # 誤差信号に表記を変える
        e = u[0].item()
        # 誤差信号を加算
        self.e_sum += e
        y = self.kp*e + self.ki*self.e_sum*self.Ts + self.kd*(e-self.prev_e)/self.Ts

        # 1ステップ前の誤差信号を保存
        self.prev_e = e
        return (y, )