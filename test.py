import numpy as np
from scipy import signal
from control import place
import IdentificationTools as ident
from scipy.signal import max_len_seq
import matplotlib.pyplot as plt

seq, state = max_len_seq(7)
u = 2*seq - 1
u = np.repeat(u, 10)

plt.plot(u)
plt.show()