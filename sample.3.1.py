import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import IdentificationTools as ident


"""
離散時間伝達関数と対応する状態空間表現のステップ応答が一致するか確認
簡単なプログラムだからchatGPTにプログラムは任せた
→ひとまず大丈夫そう．
"""

# --- ステップ入力 ---
N = 60
u = np.ones(N)

# --- QtransferFunc の応答 ---
plant:ident.QtransferFunc = ident.QtransferFunc(num=np.array([0.2]), den=np.array([-0.8]), delay=0, predict=False)
y_tf = np.zeros(N)
for k in range(N):
    y_tf[k] = plant(np.array([[u[k]]], dtype=float))[0] 

# --- 状態空間(tf2ss)の応答 ---
A, B, C, D = signal.tf2ss([0.2], [1.0, -0.8])

x = np.zeros((A.shape[0],), dtype=float)
y_ss = np.zeros(N)

for k in range(N):
    y_ss[k] = (C @ x + D * u[k]).item()
    x = (A @ x + (B.flatten() * u[k]))

# --- 比較 ---
diff = y_tf - y_ss
print("max |diff| =", np.max(np.abs(diff)))

plt.figure()
plt.plot(y_tf, label="QtransferFunc")
plt.plot(y_ss, "--", label="tf2ss state-space")
plt.grid(True)
plt.xlabel("k")
plt.ylabel("y")
plt.legend()
plt.show()
