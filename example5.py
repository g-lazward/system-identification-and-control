import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


# システム同定データの読み込み
file = np.loadtxt("identification/closed_loop_identification_simulation/IO-data.csv", delimiter=",", skiprows=1)

time = file[:, 0]
reference = file[:, 1]
input = file[:, 2]
output = file[:, 3]

fig = plt.figure(figsize=(8, 6))
ax_top = fig.add_subplot(2, 1, 1)
ax_top.plot(time, reference)
ax_top.plot(time, output)
ax_bottom = fig.add_subplot(2, 1, 2)
ax_bottom.plot(time, input)
plt.show()

