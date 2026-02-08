import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import identctrl.identification as ident
from pprint import pprint


# システム同定データの読み込み
file = np.loadtxt("./examples/data/IO-data.csv", delimiter=",", skiprows=1)

# 各信号の抽出
time = file[:, 0]
reference = file[:, 1]
input = file[:, 2]
output = file[:, 3]

# モデルの初期化
ru_model: ident.QtransferFunc = ident.QtransferFunc(num=np.array([0., 0.,]), den=np.array([0., 0., 0.,]), delay=0, predict=True)
inv_ru_noise_model: ident.QtransferFunc = ident.QtransferFunc(num=np.array([1.]), den=np.array([0.3]), delay=0, predict=True)

# パラメータ初期値の設定
init_param = np.vstack([ru_model.parameter, inv_ru_noise_model.parameter]).reshape(-1)
print(f"paramter initial value: {init_param}")
param_history = []

# モデル構造の表示
print(f"r->uを表すモデル：{ru_model}")
print(f"ノイズ：{inv_ru_noise_model}")

# 残差関数の定義
def residuals(param, input, output):
    param_history.append(param.copy())
    ru_model.parameter = param[0:ru_model.parameter.shape[0]].reshape(-1, 1)
    inv_ru_noise_model.parameter = param[ru_model.parameter.shape[0]:].reshape(-1, 1)

    # 最初の方のステップはリグレッサーの初期条件の影響を受けるので無視する
    waste_num = 10

    # リグレッサーの状態リセット(繰り返し呼び出し時に前回の状態が残らないように)
    ru_model.reset()
    inv_ru_noise_model.reset()
    
    predicted_error = np.zeros(len(output)-1, dtype=float)
    for step in range(len(output)-1):
        
        error = output[step] - ru_model(np.array([[input[step]]], dtype=float))[0]
        predicted_error[step] = inv_ru_noise_model(np.array([[error]], dtype=float))[0]
    
    return predicted_error[waste_num:]

# 1 step-ahead-predictionの式で最適化によってH^{-1}の分子多項式を0️にされるのを防ぐために，パラメータの下限・上限を設定
lb = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.1])
ub = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, 5.0])

result = least_squares(
    residuals,
    x0=init_param,
    bounds=(lb, ub),
    args=(reference, input),
    method="trf",
    verbose=1,
    max_nfev=50,
    ftol=1e-4, xtol=1e-4, gtol=1e-4
)


ru_model.parameter = result.x[0:ru_model.parameter.shape[0]].reshape(-1, 1)
# pprint(f"推定r->uパラメータ: {ru_model.parameter})

print(f"推定r->u伝達関数: {ru_model}")

# 予測入力のプロット
predicted_input = np.zeros(len(output), dtype=float)
for step in range(len(output)):
    predicted_input[step] = ru_model(np.array([[reference[step]]], dtype=float))[0]


# fig = plt.figure(figsize=(8, 6))
# ax_top = fig.add_subplot(2, 1, 1)
# ax_top.plot(time, reference, label="reference")
# ax_top.plot(time, output, label="output")
# ax_top.set_ylabel("output [-]")
# ax_top.grid()
# ax_top.legend()
# ax_bottom = fig.add_subplot(2, 1, 2)
# ax_bottom.plot(time, input, label="input")
# ax_bottom.plot(time, predicted_input, label="predicted input")
# ax_bottom.set_xlabel("step [-]")
# ax_bottom.set_ylabel("input [-]")
# ax_bottom.grid()
# ax_bottom.legend()
# fig.savefig("./examples/figure/example5_ident.svg", format="svg", transparent=True)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(1, 1, 1)
# param_history = np.array(param_history)
# ax.plot(param_history[:, 0], label="plant num")
# ax.plot(param_history[:, 1], label="plant den")
# ax.plot(param_history[:, 2], label="inv noise num")
# ax.plot(param_history[:, 3], label="inv noise den")
# ax.set_xlabel("iteration [-]")
# ax.set_ylabel("parameter [-]")
# ax.grid()
# ax.legend()
# fig.savefig("./examples/figure/example5_parameters.svg", format="svg", transparent=True)

# plt.show()



# ------------------------- step 2 ------------------------------ #
# モデルの初期化
# uy_model: ident.QtransferFunc = ident.QtransferFunc(num=np.array([1., 0.8, 0.12]), den=np.array([1.8, 1, 0.15,]), delay=0, predict=True)
uy_model: ident.QtransferFunc = ident.QtransferFunc(num=np.array([0., 0., 0.]), den=np.array([0., 0., 0.]), delay=0, predict=True)
inv_uy_noise_model: ident.QtransferFunc = ident.QtransferFunc(num=np.array([1.]), den=np.array([0.3]), delay=0, predict=True)

# パラメータ初期値の設定
init_param = np.vstack([uy_model.parameter, inv_uy_noise_model.parameter]).reshape(-1)
print(f"paramter initial value: {init_param}")
param_history = []

# モデル構造の表示
print(f"r->uを表すモデル：{uy_model}")
print(f"ノイズ：{inv_uy_noise_model}")



# 残差関数の定義
def residuals(param, input, output):
    param_history.append(param.copy())
    uy_model.parameter = param[0:uy_model.parameter.shape[0]].reshape(-1, 1)
    inv_uy_noise_model.parameter = param[uy_model.parameter.shape[0]:].reshape(-1, 1)

    # 最初の方のステップはリグレッサーの初期条件の影響を受けるので無視する
    waste_num = 10

    # リグレッサーの状態リセット(繰り返し呼び出し時に前回の状態が残らないように)
    uy_model.reset()
    inv_uy_noise_model.reset()
    
    predicted_error = np.zeros(len(output)-1, dtype=float)
    for step in range(len(output)-1):
        
        error = output[step] - uy_model(np.array([[input[step]]], dtype=float))[0]
        predicted_error[step] = inv_uy_noise_model(np.array([[error]], dtype=float))[0]
    
    return predicted_error[waste_num:]

# 1 step-ahead-predictionの式で最適化によってH^{-1}の分子多項式を0️にされるのを防ぐために，パラメータの下限・上限を設定
lb = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.1])
ub = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, 5.0])

result = least_squares(
    residuals,
    x0=init_param,
    bounds=(lb, ub),
    args=(predicted_input, output),
    method="trf",
    verbose=1,
    max_nfev=200,
    ftol=1e-6, xtol=1e-6, gtol=1e-6
)


uy_model.parameter = result.x[0:uy_model.parameter.shape[0]].reshape(-1, 1)
# pprint(f"推定r->uパラメータ: {ru_model.parameter})

print(f"推定u->y伝達関数: {uy_model}")

# 予測出力のプロット(もともとの不安定系の出力だから発散する）
predicted_output = np.zeros(len(output), dtype=float)
for step in range(len(output)):
    predicted_output[step] = uy_model(np.array([[predicted_input[step]]], dtype=float))[0]


fig = plt.figure(figsize=(8, 6))
ax_top = fig.add_subplot(2, 1, 1)
ax_top.plot(time, reference, label="reference")
ax_top.plot(time, output, label="output")
ax_top.plot(time, predicted_output, label="predicted output")
ax_top.set_ylabel("output [-]")
ax_top.grid()
ax_top.legend()
ax_bottom = fig.add_subplot(2, 1, 2)
ax_bottom.plot(time, input, label="input")
ax_bottom.plot(time, predicted_input, label="predicted input")
ax_bottom.set_xlabel("step [-]")
ax_bottom.set_ylabel("input [-]")
ax_bottom.grid()
ax_bottom.legend()
fig.savefig("./examples/figure/example5_ident2.svg", format="svg", transparent=True)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
param_history = np.array(param_history)
ax.plot(param_history)
ax.set_xlabel("iteration [-]")
ax.set_ylabel("parameter [-]")
ax.grid()
ax.legend()
fig.savefig("./examples/figure/example5_parameters2.svg", format="svg", transparent=True)

plt.show()


