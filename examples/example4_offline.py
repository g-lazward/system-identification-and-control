import numpy as np
import matplotlib.pyplot as plt
import identctrl.identification as ident
import identctrl.plotter as plotter
from scipy import signal
from pprint import pprint


# @brief 矩形波を生成
# @args t: 時間配列，T: 周期
def square_signal(t, T)->np.ndarray:
    return (np.sin(2 * np.pi * t / T) > 0).astype(float)



time: np.ndarray = np.arange(0, 500, dtype=float)
t1, t2, t3, t4 = 20, 5, 2.5, 3.333
f1, f2, f3, f4 = 1/t1, 1/t2, 1/t3, 1/t4
print(f"f1: {f1}, f2: {f2}, f3: {f3}, f4: {f4}")
signal1: np.ndarray = 0.2*np.sin(2*np.pi*time/t1)
signal2: np.ndarray = 0.5*np.sin(2*np.pi*time/t2)
signal3: np.ndarray = 0.9*np.sin(2*np.pi*time/t3)
signal4: np.ndarray = np.sin(2*np.pi*time/t4)

u = signal1 + signal3 + signal4


plant:ident.QtransferFunc = ident.QtransferFunc(num=np.array([1, -0.1, -0.2]), den=np.array([-0.1, -0.44, 0.084]), delay=0, predict=True)
regressor_matrix: np.ndarray = np.full((len(plant.den)+len(plant.num), len(time)), np.nan, dtype=float)
print(f"regressor_matrix shape: {regressor_matrix.shape}")

y = np.full_like(time, np.nan, dtype=float)
y [0] = 0.0

for step in range(len(time)-1):
    regressor_matrix[:, step] = plant.regressor.flatten()
    y[step+1] = plant(np.array([[u[step]]], dtype=float))[0] + np.random.normal(0, 0.01)
    # y[step+1] = plant(np.array([[u[step]]], dtype=float))[0]

# 最後のステップの回帰ベクトルを格納
regressor_matrix[:, -1] = plant.regressor.flatten()

print(f"true parameter: \n{plant.parameter.reshape(-1, 1)}")
print(f"estimated parameter: \n{np.linalg.inv(regressor_matrix @ regressor_matrix.T) @ regressor_matrix @ y.reshape(-1, 1)}")



### 解析 ###
reg: np.ndarray = np.full((len(plant.den)+len(plant.num), len(time)), np.nan, dtype=float)
for step in range(len(time)):
    phiphiT = regressor_matrix[:, step].reshape(-1, 1) @ regressor_matrix[:, step].reshape(1, -1)
    reg[:, step] = np.abs(np.linalg.eig(phiphiT)[0])**2

# reg_fig = plt.figure(figsize=(8, 6))
# reg_ax = reg_fig.add_subplot(1, 1, 1)
# for i in range(len(plant.den)+len(plant.num)):
#     if (i < len(plant.den)):
#         reg_ax.plot(time, reg[i, :], label=f"den {i}", marker='.')
#     else:
#         reg_ax.plot(time, reg[i, :], label=f"num {i-len(plant.den)}", marker='.')
#     print(f"reg {i}'s max: {np.max(reg[i, :])}")
# reg_ax.legend()
# reg_ax.grid()
# plt.show()

print(f"phiphi eig: {np.linalg.eig(regressor_matrix @ regressor_matrix.T)[0]}")
print(f"condition number: {np.linalg.cond(regressor_matrix @ regressor_matrix.T)}")



## グラフ表示
figure = plt.figure(figsize=(8, 6))
ax = figure.add_subplot(1, 1, 1)
ax.plot(time, u, label=f'input', marker='.')
ax.plot(time, y, label='output', marker='.')
ax.set_xlabel("step [-]")
ax.set_ylabel("output [-]")
ax.grid()
ax.legend()
# plt.show()



f, P = plotter.pSpectrum(u, fs=1.0, window=None)
f2, P2 = plotter.pSpectrum(y, fs=1., window=None)
plt.figure(figsize=(8, 6))
plt.plot(f, P, marker='.', label='input')
plt.plot(f2, P2, marker='.', label='output')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power Spectrum [-]") 
plt.grid(True)
plt.legend()
# plt.show()