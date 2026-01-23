import numpy as np
from numpy.typing import NDArray
from typing import Union
from dataclasses import dataclass
import IdentificationTools as ident

from cvxopt import matrix
from cvxopt.solvers import qp


FloatArray1D = NDArray[np.floating]
FloatArray2D = NDArray[np.floating]
Scalar = Union[float, np.floating] 

@dataclass
# Linear Discrete State-Space model
class LDss:
    A: FloatArray2D
    B: FloatArray2D
    C: FloatArray2D


class MPC:

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
        self.system: LDss = system

        
        self.Hp = int(Hp)
        self.Hu = int(Hu)
        assert self.Hp >= self.Hu, "Hp >= Huを満たしていません"

        self.Hw = int(Hw)

        self.Q = Q
        self.R = R
        self.L = L

        # システムの次元
        self.system_dim:int = self.system.A.shape[0]
        # 入力のベクトルサイズ
        self.input_dim:int = self.system.B.shape[1]
        # 出力のベクトルサイズ
        self.output_dim:int = self.system.C.shape[0]

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
        
        P = matrix(self.P)
        q = matrix(2 * self.Theta.T @ self.Q @ (self.Psi @ state + self.Upsilon @ prev_u - T))

        solution = qp(P=P, q=q)
        delta_U = np.array(solution["x"]).reshape(self.input_dim*self.Hu, 1)

        return delta_U[:self.input_dim, :]

    ## @brief 状態観測器（Luenberger観測器)
    # @args state: 状態ベクトル, u: 入力ベクトル，y: 出力ベクトル
    def _observeState(self, state, u, y):
        
        assert state.shape == (self.system_dim, 1)
        assert u.shape == (self.input_dim, 1)
        assert y.shape == (self.output_dim, 1)

        next_state = (self.system.A - self.L @ self.system.C) @ state + self.system.B @ u + self.L @ y
        return next_state

    ## @brief 呼び出して使用するメイン処理
    # @args T: 参照信号ベクトル(出力信号が複数ある場合は行列)，y: 制御対象の出力(状態観測器で使用)
    # @output: 制御対象への入力u(k|k)，推定状態ベクトル\hat{x}(k+1|k)
    def forward(self, T:FloatArray2D, y: FloatArray2D)->dict:
        assert T.shape == (self.output_dim*self.Hp, 1), "Tは(output_dim*Hp, 1)のサイズで与えてください"
        
        # 入力の差分を計算
        du = self._calcInput(self.state, self.u, T)
        self.u = self.u + du

        # 次のステップの推定状態ベクトルの計算(時間ステップに注意)
        # \hat{x}(k+1|k) = observer(\hat{x}(k|k-1), u(k), y(k))
        self.state = self._observeState(self.state, self.u, y)
        
        return {"u": self.u, "state": self.state}
