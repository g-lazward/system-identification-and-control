import IdentificationTools as ident
import ControllerTools as ctrl
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# @brief 矩形波を生成
# @args t: 時間配列，T: 周期
def square_signal(t, T)->np.ndarray:
    return (np.sin(2 * np.pi * t / T) > 0).astype(int)


if __name__ == "__main__":
    # 制御対象
    plant:ident.QtransferFunc = ident.QtransferFunc(num=np.array([0.2]), den=np.array([-0.8]), delay=0, predict=True)

    # scipy用制御対象
    b = [0.2]
    a = [1.0, -0.8]
    zi = signal.lfilter_zi(b, a) * 0.0

    # コントローラー
    controller:ctrl.SimplePIDController = ctrl.SimplePIDController(kp=0.5, ki=2, kd=0)
    # scipy用のコントローラー
    controller_scipy:ctrl.SimplePIDController = ctrl.SimplePIDController(kp=0.5, ki=2, kd=0)

    # 時間配列
    time = np.arange(0, 100, 1)

    ## ----- 選べる参照信号 -----
    # step
    reference = np.zeros_like(time)
    reference[10:-1] = 1

    # square
    # reference = square_signal(time, T=40)

    # sin
    # reference = np.sin(2*np.pi/20*time)
    ## ------------------

    error = np.zeros_like(time, dtype=float)
    error_scipy = np.zeros_like(time, dtype=float)
    output = np.zeros_like(time, dtype=float)
    output_scipy = np.zeros_like(time, dtype=float)
    input = np.zeros_like(time, dtype=float)
    input_scipy = np.zeros_like(time, dtype=float)

    for step in range(len(time)-1):
        # 自作プログラムの閉ループ系
        error[step] = reference[step] - output[step]
        input[step] = controller(np.array([[error[step]]], dtype=float))[0]
        output[step+1] = plant(np.array([[input[step]]], dtype=float))[0]
    
        
        # scipyを使った閉ループ系
        error_scipy[step] = reference[step] - output_scipy[step]
        input_scipy[step] = controller_scipy(np.array([[error_scipy[step]]], dtype=float))[0]
        next_output, zi = signal.lfilter(b, a, [input_scipy[step]], zi=zi)
        output_scipy[step+1] = next_output[0]


    ## ----- 以下プロット -----
    fig = plt.figure(figsize=(8, 6))

    ax_top = fig.add_subplot(3, 1, 1)
    ax_top.set_ylabel("output [-]")
    ax_top.plot(time, reference, marker='.', label='reference', linestyle='None')
    ax_top.plot(time, output, marker='.', label='plant(my program)')
    ax_top.plot(time, output_scipy, label='plant(scipy)')
    plt.legend()
    plt.grid()

    ax_center = fig.add_subplot(3, 1, 2)
    ax_center.set_ylabel('error [-]')
    ax_center.plot(time, error, marker='.')
    plt.grid()

    ax_bottom = fig.add_subplot(3, 1, 3)
    ax_bottom.set_xlabel('time [step]')
    ax_bottom.set_ylabel('input [-]')
    ax_bottom.plot(time, input, marker='.')
    plt.grid()

    # plt.savefig("step.svg", format="svg", transparent=True)
    plt.show()
    ## ----- ----------- -----