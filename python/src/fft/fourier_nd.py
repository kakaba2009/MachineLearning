import numpy as np
import pylab as plt
import pandas as pd
from numpy import fft
import src.mylib.mfile as mfile

def bandpass_filter(x, freq, frequency_of_signal=0, band=0.1):
    if (frequency_of_signal - band) < abs(freq) < (frequency_of_signal + band):
        return x
    else:
        return 0

d1 = mfile.loadClose('JPY=X', '../db/forex.db')
d2 = mfile.loadClose('GBP=X', '../db/forex.db')
ds = pd.concat([d1, d2], axis=1, join='inner')
x = ds.values
for i in range(1):
    x = x[i:]
    N = len(x)
    spectrum = fft.fftn(x)
    feq = fft.fftfreq(N)            # frequencies
    ampli = np.absolute(spectrum)   # amplitude
    phase = np.angle(spectrum)      # phase
    #print(phase)
    index = np.argsort(-ampli, axis = 0)
    sfreq = feq[index]
    sampl = ampli[index]
    #print(sampl[1:10])
    #sfreq = np.where(sfreq > 0)
    #big = list(zip(*sfreq))
    print(sfreq[1:10] * N)
    plt.plot(sfreq * N, 'o')
#F_filtered = np.asanyarray([bandpass_filter(x, freq) for x, freq in zip(spectrum, feq)])
#filtered_signal = np.fft.ifft(F_filtered)

#plt.semilogy(feq[1:], ampli[1:]), 'o') #zero feq is very large
#plt.semilogy(ampli[1:])
plt.legend()
plt.show()
