import numpy as np
import matplotlib.pyplot as plt
import identctrl.identification as ident
from pprint import pprint
import copy


time: np.ndarray = np.arange(0, 1000, dtype=float)
u: np.ndarray = 0.5*np.sin(2*np.pi*time*0.025) + 0.3*np.sin(2*np.pi*time*0.04) + 0.7*np.sin(2*np.pi*time*0.056)

plant: ident.QtransferFunc = ident.QtransferFunc(num=np.array([1, -0.1, -0.2]), den=np.array([-0.1, -0.44, 0.084]), delay=0, predict=True)
model: ident.QtransferFunc = ident.QtransferFunc(num=np.array([0, 0, 0], dtype=float), den=np.array([0, 0, 0], dtype=float), delay=0, predict=True)

# パラメータ推定器の初期パラメータベクトル
init_parameter: np.ndarray = model.parameter

# 初期共分散行列
init_P: np.ndarray = np.diag(np.ones_like(init_parameter).reshape(-1))

# RLSの忘却係数
discount: float = 0.98

# 通常のRLS
estimator: ident.RLS = ident.RLS(theta_init=init_parameter, P_init=init_P, discount=discount)
parameter_history: np.ndarray = np.full_like(np.zeros((len(init_parameter), len(time))), np.nan)


# MatrixForgettingRLS(通常のRLSと等価として設定)
# B = 1/np.sqrt(discount)*np.eye(len(init_parameter))
MFRLS_estimator: ident.MFRLS = ident.MFRLS(theta_init=init_parameter, P_init=init_P, discount=discount)
MFRLS_parameter_history: np.ndarray = np.full_like(np.zeros((len(init_parameter), len(time))), np.nan)
MFRLS_model: ident.QtransferFunc = copy.deepcopy(model)


y = np.full_like(time, np.nan, dtype=float)
y_RLS = np.full_like(time, np.nan, dtype=float)
y_MFRLS = np.full_like(time, np.nan, dtype=float)
y[0], y_RLS[0], y_MFRLS[0] = 0.0, 0.0, 0.0


for step in range(len(time)-1):
    # システムのパラメータを途中で変更
    if step == 800:
        num, den = np.array([1, 0.1, -0.12], dtype=float), np.array([-0.6, 0.4, -0.3], dtype=float)
        plant.parameter = np.concatenate([den, num]).reshape(-1, 1)
        # print("plant parameter", plant.parameter.flatten())

    # パラメータ推定
    estimated_parameter = estimator.forward([model.regressor, y[step]])[0]
    MFRLS_estimated_parameter = MFRLS_estimator.forward([model.regressor, y[step]])[0]

    # 推定パラメータの保存
    parameter_history[:, step] = estimated_parameter.reshape(-1)
    MFRLS_parameter_history[:, step] = MFRLS_estimated_parameter.reshape(-1)

    # モデルに推定パラメータを設定
    model.parameter = estimated_parameter
    MFRLS_model.parameter = MFRLS_estimated_parameter
    
    
    
    y_RLS[step+1] = model.forward([u[step]])[0]
    y_MFRLS[step+1] = MFRLS_model.forward([u[step]])[0]
    

    y[step+1] = plant.forward([u[step]])[0] #+ 0.01*np.random.randn()



clipping = 750
fig = plt.figure(figsize=(8, 6))
ax_top = fig.add_subplot(2, 1, 1)
ax_top.plot(time[clipping:], y[clipping:], label="plant")
ax_top.plot(time[clipping:], y_RLS[clipping:], label="model")
ax_top.plot(time[clipping:], y_MFRLS[clipping:], label="MFRLS model")
ax_top.set_ylabel("Output [-]")
ax_top.legend()
ax_top.grid()
ax_bottom = fig.add_subplot(2, 1, 2)
ax_bottom.plot(time, u)
ax_bottom.set_xlabel("Time [s]")
ax_bottom.set_ylabel("Input [-]")
ax_bottom.grid()



clipping = 750
parameter_fig = plt.figure(figsize=(8, 6))
ax = parameter_fig.add_subplot(1, 1, 1)
for i in range(len(plant.parameter)):
    ax.axhline(plant.parameter[i])

for i in range(len(parameter_history[:, 0])):

    line, = ax.plot(time, parameter_history[i, :], label=f"parameter: {i+1}")
    
    ax.plot(
        time,
        MFRLS_parameter_history[i, :],
        label=f"MFRLS parameter: {i+1}",
        linestyle="dashed",
        color=line.get_color()
    )
ax.set_ylabel("Parameter [-]")
ax.set_xlabel("Time [-]")
ax.set_xlim(clipping, time[-1])
# ax.set_ylim(-2, 2)
ax.grid()
ax.legend(ncol=2)

fig_parameter_norm = plt.figure(figsize=(8, 6))
ax = fig_parameter_norm.add_subplot(1, 1, 1)
parameter_norm = np.linalg.norm(parameter_history, axis=0)
MFRLS_parameter_norm = np.linalg.norm(MFRLS_parameter_history, axis=0)
ax.plot(time, parameter_norm, label="RLS parameter norm")
ax.plot(time, MFRLS_parameter_norm, label="MFRLS parameter norm", linestyle="dashed")
ax.set_xlabel("Time [-]")
ax.set_ylabel("Parameter norm [-]")
ax.set_xlim(clipping, time[-1])
ax.grid()
ax.legend()


plt.show()