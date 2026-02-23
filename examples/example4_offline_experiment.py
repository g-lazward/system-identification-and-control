import numpy as np
import matplotlib.pyplot as plt
import identctrl.identification as ident
import identctrl.plotter as plotter
from scipy import signal
from pprint import pprint


max_step = 10
time: np.ndarray = np.arange(0, max_step, dtype=float)
t1, t2, t3, t4 = 19, 5, 2.2, 3.333
f1, f2, f3, f4 = 1/t1, 1/t2, 1/t3, 1/t4
print(f"f1: {f1}, f2: {f2}, f3: {f3}, f4: {f4}")
signal1: np.ndarray = np.sin(2*np.pi*time/t1)
signal2: np.ndarray = np.sin(2*np.pi*time/t2)
signal3: np.ndarray = np.sin(2*np.pi*time/t3)
signal4: np.ndarray = np.sin(2*np.pi*time/t4)

u = np.full((len(time), 4), np.nan, dtype=float)
u[:, 0] = signal1
u[:, 1] = signal1 + signal2
u[:, 2] = signal1 + signal2 + signal3
u[:, 3] = signal1 + signal2 + signal3 + signal4
plant:ident.QtransferFunc = ident.QtransferFunc(num=np.array([1, -0.1, -0.2]), den=np.array([-0.1, -0.44, 0.084]), delay=0, predict=True)

estimated_parameter = np.full((100, 4), np.nan, dtype=float)

for i in range(100):
    for case in range(4):
        plant.reset()
        regressor_matrix: np.ndarray = np.full((len(plant.den)+len(plant.num), len(time)), np.nan, dtype=float)

        y = np.full_like(time, np.nan, dtype=float)
        y [0] = 0.0

        for step in range(len(time)-1):
            regressor_matrix[:, step] = plant.regressor.flatten()
            y[step+1] = plant(np.array([[u[step, case]]], dtype=float))[0] + np.random.normal(0, 0.1)
            # y[step+1] = plant(np.array([[u[step, case]]], dtype=float))[0]
        # 最後のステップの回帰ベクトルを格納
        regressor_matrix[:, -1] = plant.regressor.flatten()

        estimated_parameter[i, case] = np.linalg.norm(np.linalg.inv(regressor_matrix @ regressor_matrix.T) @ regressor_matrix @ y.reshape(-1, 1).flatten() - plant.parameter.reshape(-1, 1).flatten())
        


box_fig = plt.figure(figsize=(8, 6))
box_ax = box_fig.add_subplot(1, 1, 1)
box_ax.boxplot(estimated_parameter)
box_ax.set_ylabel("Parameter Estimation Error")
box_ax.set_xticklabels(["case 1", "case 2", "case 3", "case 4"])
box_ax.text(0.95, 0.95, f"n={100}, max_step={max_step}", transform=plt.gca().transAxes, ha="right", va="top", bbox=dict(boxstyle="round", fc="white", ec="gray"))
box_ax.set_ylim([-0.5, 25.5])
box_ax.grid()
box_fig.savefig(f"examples/figure/example4_experiment1_{max_step}.svg", transparent=True, format="svg")

