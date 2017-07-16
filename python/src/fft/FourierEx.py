import numpy as np
import pylab as plt
from numpy import fft
import src.mylib.mlstm as mlstm

def bandpass_filter(x, freq, frequency_of_signal=0, band=0.1):
    if (frequency_of_signal - band) < abs(freq) < (frequency_of_signal + band):
        return x
    else:
        return 0

ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 9000)
x = ds[['Close']].values
x = x.flatten()
L = []
for i in range(1000):
    x = x[i:]
    N = len(x)
    spectrum = fft.fft(x)
    feq = fft.fftfreq(N)            # frequencies
    ampli = np.absolute(spectrum)   # amplitude
    phase = np.angle(spectrum)      # phase
    #print(phase)
    index = np.argsort(-ampli)
    sfreq = feq[index]
    sampl = ampli[index]
    #sfreq = np.where(sfreq > 0)
    #big = list(zip(*sfreq))
    print(sfreq[1:10] * N)
    #plt.semilogy(sfreq * N, sampl, 'o')
#F_filtered = np.asanyarray([bandpass_filter(x, freq) for x, freq in zip(spectrum, feq)])
#filtered_signal = np.fft.ifft(F_filtered)

#plt.semilogy(feq[1:], ampli[1:]), 'o') #zero feq is very large
#plt.semilogy(ampli[1:])
plt.legend()
plt.show()
