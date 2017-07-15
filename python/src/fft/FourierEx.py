import numpy as np
import pylab as plt
from numpy import fft
import src.mylib.mlstm as mlstm

ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 1024)
x = ds[['Close']].values
x = x.flatten()
#t = np.arange(256)
#x = np.sin(t)

n = x.size
spectrum = fft.fft(x)
feq = fft.fftfreq(n)            # frequencies

ampli = np.absolute(spectrum)   # amplitude
phase = np.angle(spectrum)      # phase
print(phase)

plt.plot(ampli)
plt.legend()
plt.show()
