import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from matplotlib import style

def m_spectrum(df, feq, column):
    Fs = feq
    
    t = df.index
    s = df[column]
    
    plt.subplot(3, 2, 1)
    plt.plot(t, s)
    
    plt.subplot(3, 2, 3)
    plt.magnitude_spectrum(s, Fs=Fs)
    
    plt.subplot(3, 2, 4)
    plt.magnitude_spectrum(s, Fs=Fs, scale='dB')
    
    plt.subplot(3, 2, 5)
    plt.angle_spectrum(s, Fs=Fs)
    
    plt.subplot(3, 2, 6)
    plt.phase_spectrum(s, Fs=Fs)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
def m_xcorr(X, Y):
    x, y = X, Y
    
    len = x.size
    
    #matplot lib max limitation is 3335
    if(len >= 3335):
        len = 3335

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.xcorr(x, y, usevlines=False, normed=True, lw=2)
    ax1.grid(True)
    ax1.axhline(0, color='blue', lw=2)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
def m_acorr(X, maxlag):
    fig = plt.figure()
    
    x = X
    
    len = x.size
    
    #matplot lib max limitation is 1038
    if(maxlag >= 1000):
        maxlag = 1000
    
    ax2 = fig.add_subplot(212)
    ax2.acorr(x, usevlines=False, normed=False, maxlags=maxlag, lw=1, linestyle='solid')
    ax2.grid(True)
    ax2.axhline(0, color='blue', lw=2)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
