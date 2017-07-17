import numpy as np
import pylab as plt
from numpy import fft
import src.mylib.mlstm as mlstm

def bandpass_filter(x, freq, frequency_of_signal=0, band=0.1):
    if (frequency_of_signal - band) < abs(freq) < (frequency_of_signal + band):
        return x
    else:
        return 0

ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 3500)
y = ds[['Close']].values
y = y.flatten()
for k in range(5, 30):
    for i in range(0, 3000, 1):
        x = y[i:i+k]
        N = len(x)
        spectrum = fft.fft(x)
        feq = fft.fftfreq(N)            # frequencies
        ampli = np.absolute(spectrum)   # amplitude
        phase = np.angle(spectrum)      # phase
        #print(phase)
        index = np.argsort(-ampli)
        sfreq = feq[index]
        sampl = ampli[index]
        sfreq = sfreq[sfreq > 0]
        #big = list(zip(*sfreq))
        if sfreq[0] * N >= k*0.5 - 1:
            print(N, sfreq[:2] * N)
        #plt.semilogy(ampli, 'o')
        #F_filtered = np.asanyarray([bandpass_filter(x, freq) for x, freq in zip(spectrum, feq)])
        #filtered_signal = np.fft.ifft(F_filtered)

        #plt.semilogy(feq[1:], ampli[1:]), 'o') #zero feq is very large
        #plt.semilogy(ampli[1:])
        #plt.title(str(N))
        #plt.legend()
        #plt.show()