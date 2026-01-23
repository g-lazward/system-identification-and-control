import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import IdentificationTools as ident

"""
自作したシフトオペレータを用いたSISO離散時間伝達関数と
scipyの（Z変換を用いた）SISO離散伝達関数の等価性を検証するプログラム
"""

# 例: G(z) = (b0 + b1 z^-1 + ...)/(a0 + a1 z^-1 + ...)
b = [0, 0.2]      # numerator
a = [1.0, -0.8]     # denominator
dt = 1            # サンプリング周期

# 上記のZ変換方式の伝達関数のパラメータををシフトオペレータを用いた伝達関数に変換して渡す
system = ident.QtransferFunc(num=np.array([0.2]), den=np.array([-0.8]), predict=False)

sample_num = 100
u = np.ones(sample_num)
output = np.zeros_like(u)
handwrite_output = np.zeros_like(u)

for step in range(len(u)-1):
    
    output[step] = system(np.array([[u[step]]], dtype=float))[0]
    handwrite_output[step+1] = 0.2*u[step] + 0.8*handwrite_output[step]

sci_system = signal.dlti(b, a, dt=dt)
t, y = signal.dstep(sci_system, n=sample_num)
t = np.asarray(t).squeeze()
y = np.asarray(y).squeeze()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.plot(t, y, label="scipy")
ax.plot(t, output, label="my_program")
ax.plot(t, handwrite_output, label="handwriting")
plt.grid()
plt.legend()
plt.savefig("step.svg", format="svg", transparent=True)
plt.show()