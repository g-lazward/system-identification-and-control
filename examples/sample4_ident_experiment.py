import numpy as np
import matplotlib.pyplot as plt
from identctrl import identification as ident
from scipy import signal
from control import place
from sample2 import square_signal

import os

import csv

# 不安定システム
un_stable_system = ident.QtransferFunc(num=np.array([1, 0.8, 0.12]), den=np.array([1.9, 0.96, 0.144]), predict=True)
A, B, C, D = signal.tf2ss([0, 1, 0.8, 0.12], [1, 1.9, 0.96, 0.144])
print("---- unstable system matrices ----")
print(A, B, C, D)
print("-------------------------")

# 拡大系を構成
Ae = np.block([
    [A, np.zeros((A.shape[0], 1))],
    [-C, 1]
])
Be = np.vstack([B, 0])
Ee = np.vstack([
    np.zeros((A.shape[0], 1)),
    1
])
Ce = np.hstack([C, np.zeros((1, 1))])
print("----拡大系-----")
print("Ae:",Ae)
print("Be", Be)
print("Ee:", Ee)
print("Ce:", Ce)
print("---------------")

# 極配置
poles = [0.2, 0.25, 0.3, 0.35]
K = place(Ae, Be, poles)
# 状態フィードバックゲイン
print(K)


# 時間配列
time = np.arange(0, 635, 1)
# square
# reference = square_signal(time, T=50)

# m-sequence
seq, state = signal.max_len_seq(7)
u = 2*seq - 1
u = np.repeat(u, 5)
reference = u


input = np.zeros_like(time, dtype=float)
output = np.zeros_like(time, dtype=float)
system_state = np.zeros((len(time), Ae.shape[0]))


for step in range(len(time)-1):
    output[step] = (Ce @ system_state[step]).item()

    # 積分状態だけ先に更新
    state_temp = system_state[step].copy()
    state_temp[-1] = system_state[step, -1] + (reference[step] - output[step])

    # 入力後の積分状態で計算
    input[step] = (-K @ state_temp).item()

    # 状態更新
    system_state[step+1] = Ae @ system_state[step] + Be[:, 0] * input[step] + Ee[:, 0] * reference[step]

    # 積分状態を確定（ズレ防止）
    system_state[step+1, -1] = state_temp[-1]

fig = plt.figure(figsize=(8, 6))
ax_top = fig.add_subplot(2, 1, 1)
ax_top.plot(time, reference, label="reference", marker='.')
ax_top.plot(time, output, label="system", marker=".")
ax_top.legend()
ax_top.grid()
ax_top.set_ylabel("Output [-]")
ax_bottom = fig.add_subplot(2, 1, 2)
ax_bottom.plot(time, input)
ax_bottom.grid()
ax_bottom.set_ylabel("Input [-]")
ax_bottom.set_xlabel("Time [step]")
plt.savefig("./examples/figure/example4_ident_experiment.svg", format="svg")
# plt.show()

folder_name = "./examples/data"
output_name = "IO-data.csv"

with open(os.path.join(folder_name, output_name), "w") as file:
    writer = csv.writer(file)
    writer.writerow(["step", "reference", "input", "output"])
    for row in range(len(time)-1):
        writer.writerow([time[row], reference[row], input[row], output[row]])

