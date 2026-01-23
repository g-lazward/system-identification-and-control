import IdentificationTools as ident
import ControllerTools as cont
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

from sample2 import square_signal
from numpy.typing import NDArray
from typing import Union
FloatArray1D = NDArray[np.floating]
FloatArray2D = NDArray[np.floating]
Scalar = Union[float, np.floating]

# 制御対象
plant:ident.QtransferFunc = ident.QtransferFunc(num=np.array([0.2]), den=np.array([-0.8]), delay=0, predict=True)

# 伝達関数を実現
A, B, C, D = signal.tf2ss([0.2], [1.0, -0.8])
# 制御対象の離散時間状態空間表現
plant_ss: cont.LDss = cont.LDss(A, B, C)
print(plant_ss)

### -----モデル予測制御器(制御対象と同じモデルを与える)----- ###
Hp = 5
Hu = 3
Hw = 0
# 重み行列はひとまず，単位行列に設定（Hp，Huを変えたら自動的に生成できるように単純に...）
Q = np.diag(np.ones((Hp)))
R = np.diag(np.ones((Hu)))
# luenberger観測器の重み
L = np.array([1]).reshape(-1, 1)  # Aの固有値が0.8だから0.6くらいにしておく
mpc: cont.MPC = cont.MPC(system=plant_ss, Hp=Hp, Hu=Hu, Hw=0, Q=Q, R=R, L=L)
### -------------------------------------------------------- ###


# 時間配列
time = np.arange(0, 100, 1)

### ----- 選べる参照信号 ----- ###
# step
# reference = np.zeros_like(time)
# reference[10:-1] = 1

# square
reference = square_signal(time, T=60)

# sin
# reference = np.sin(2*np.pi/20*time)

### -------------------------- ###

# 各種信号の配列
plant_output = np.zeros_like(time, dtype=float)
plant_input = np.zeros_like(time, dtype=float)
model_predicted_state = np.zeros((len(time), mpc.system_dim))


for step in range(len(time)-Hp):

    # MPCに渡す参照信号行列の構成
    T: FloatArray2D = np.array(reference[step+1:step+Hp+1], dtype=float).reshape(-1, 1)
    y: FloatArray2D = np.array(plant_output[step], dtype=float).reshape(-1, 1) # FloatArray2Dに変換

    result = mpc(T, y)
    
    plant_input[step] = result[0].item()
    model_predicted_state[step, :] = result[1].T

    plant_output[step+1] = plant.forward(np.array([[plant_input[step]]], dtype=float))[0] 


### ----- プロット ----- ###
fig = plt.figure(figsize=(8, 6))
ax_top = fig.add_subplot(2, 1, 1)
ax_top.plot(time, reference, label="reference")
ax_top.step(time, plant_output, label="plant", where="post")
ax_top.set_ylabel("Output [-]")
ax_top.legend(loc="upper right")
ax_top.grid()
ax_bottom = fig.add_subplot(2, 1, 2)
ax_bottom.step(time, plant_input, where="post")
ax_bottom.set_xlabel("Time [step]")
ax_bottom.set_ylabel("input [-]")
ax_bottom.grid()
plt.savefig("mpc.svg", format="svg", transparent=True)
plt.show()
### -------------------- ###
