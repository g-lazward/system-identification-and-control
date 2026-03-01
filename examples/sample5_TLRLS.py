import numpy as np
import matplotlib.pyplot as plt
import identctrl.identification as ident
from pprint import pprint

time: np.ndarray = np.arange(0, 1000, dtype=float)
u: np.ndarray = 0.5*np.sin(2*np.pi*time*0.025) + 0.3*np.sin(2*np.pi*time*0.04+0.1) + 0.7*np.sin(2*np.pi*time*0.056 + 0.3)

plant: ident.QtransferFunc = ident.QtransferFunc(num=np.array([1, -0.1, -0.2]), den=np.array([-0.1, -0.44, 0.084]), delay=0, predict=True)
model: ident.QtransferFunc = ident.QtransferFunc(num=np.array([0, 0, 0], dtype=float), den=np.array([0, 0, 0], dtype=float), delay=0, predict=True)

# パラメータ推定器の初期パラメータベクトル
init_parameter: np.ndarray = model.parameter
init_P: np.ndarray = np.diag(np.ones_like(init_parameter).reshape(-1))

# estimator: ident.DFRLS = ident.DFRLS(theta_init=init_parameter, P_init=init_P, epsilon=.01, discount=0.8)
estimator: ident.TLRLS = ident.TLRLS(theta_init=init_parameter, DF_discount=0.8, EF_discount=0.8)



parameter_history: np.ndarray = np.full_like(np.zeros((len(init_parameter), len(time))), np.nan)
y = np.full_like(time, np.nan, dtype=float)
y_est = np.full_like(time, np.nan, dtype=float)
y[0], y_est[0] = 0.0, 0.0

for step in range(len(time)-1):
    print("step :", step)
    # パラメータ推定
    estimated_parameter = estimator.forward([model.regressor, y[step]])[0]
    parameter_history[:, step] = estimated_parameter.reshape(-1)
    # モデルに推定パラメータを設定
    model.parameter = estimated_parameter
    y_est[step+1] = model.forward([u[step]])[0]
    y[step+1] = plant.forward([u[step]])[0]  + 0.01*np.random.randn()
    # y[step+1] = plant.forward([u[step]])[0]


fig = plt.figure(figsize=(8, 6))
ax_top = fig.add_subplot(2, 1, 1)
ax_top.plot(time, y, label="plant")
ax_top.plot(time, y_est, label="model")
ax_top.set_ylabel("Output [-]")
ax_top.legend()
ax_top.grid()
ax_bottom = fig.add_subplot(2, 1, 2)
ax_bottom.plot(time, u)
ax_bottom.set_xlabel("Time [s]")
ax_bottom.set_ylabel("Input [-]")
ax_bottom.grid()

parameter_fig = plt.figure(figsize=(8, 6))
ax = parameter_fig.add_subplot(1, 1, 1)
for i in range(len(plant.parameter)):
    ax.axhline(plant.parameter[i])
for i in range(len(parameter_history[:, 0])):
    ax.plot(time, parameter_history[i, :], label=f"parameter: {i+1}", marker=".")
ax.set_ylabel("Parameter [-]")
ax.set_xlabel("Time [-]")
ax.grid()
ax.legend()

plt.show()