import numpy as np
import matplotlib.pyplot as plt
import identctrl.identification as ident
from pprint import pprint

time: np.ndarray = np.arange(0, 1000, dtype=float)
u: np.ndarray = 0.5*np.sin(2*np.pi*time*0.025) + 0.3*np.sin(2*np.pi*time*0.04) + 0.7*np.sin(2*np.pi*time*0.056)

plant: ident.QtransferFunc = ident.QtransferFunc(num=np.array([1, -0.1, -0.2]), den=np.array([-0.1, -0.44, 0.084]), delay=0, predict=True)
model: ident.QtransferFunc = ident.QtransferFunc(num=np.array([0, 0, 0], dtype=float), den=np.array([0, 0, 0], dtype=float), delay=0, predict=True)

# パラメータ推定器の初期パラメータベクトル
init_parameter: np.ndarray = model.parameter

# 初期共分散行列
init_P: np.ndarray = np.diag(np.ones_like(init_parameter).reshape(-1))

discount: float = 0.8
estimator: ident.RLS = ident.RLS(theta_init=init_parameter, P_init=init_P, discount=discount)


parameter_history: np.ndarray = np.full_like(np.zeros((len(init_parameter), len(time))), np.nan)
y = np.full_like(time, np.nan, dtype=float)
y_est = np.full_like(time, np.nan, dtype=float)
y[0], y_est[0] = 0.0, 0.0

for step in range(len(time)-1):

    # システムのパラメータを途中で変更
    if step == 800:
        num, den = np.array([1, 0.1, -0.12], dtype=float), np.array([-0.6, 0.4, -0.3], dtype=float)
        plant.parameter = np.concatenate([den, num]).reshape(-1, 1)
        print("plant parameter", plant.parameter.flatten())


    # パラメータ推定
    estimated_parameter = estimator.forward([model.regressor, y[step]])[0]
    parameter_history[:, step] = estimated_parameter.reshape(-1)
    # モデルに推定パラメータを設定
    model.parameter = estimated_parameter
    y_est[step+1] = model.forward([u[step]])[0]
    y[step+1] = plant.forward([u[step]])[0] + 0.01*np.random.randn()

clipping = 750
fig = plt.figure(figsize=(8, 6))
ax_top = fig.add_subplot(2, 1, 1)
ax_top.plot(time[clipping:], y[clipping:], label="plant")
ax_top.plot(time[clipping:], y_est[clipping:], label="model")
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

upper = np.linalg.cond(init_P)*np.linalg.norm(parameter_history[:, 0].reshape(-1, 1)-plant.parameter)**2

eval = np.full_like(time, np.nan)
for i in range(1, len(time)):
    eval[i] = np.linalg.norm(parameter_history[:, i].reshape(-1, 1)-plant.parameter)**2

property_fig = plt.figure(figsize=(8, 6))
property_ax = property_fig.add_subplot(1, 1, 1)
property_ax.plot(time, eval, marker=".")
property_ax.axhline(upper)
property_ax.set_xlabel("Time [-]")
property_ax.set_ylabel("Parameter Estimation Error [-]")
property_ax.grid()


fig.savefig(f"examples/figure/example5_RLS_discount{discount}.svg", transparent=True, format="svg")
parameter_fig.savefig(f"examples/figure/example5_RLS_parameter_discount{discount}.svg", transparent=True, format="svg")
property_fig.savefig(f"examples/figure/example5_RLS_property_discount{discount}.svg", transparent=True, format="svg")