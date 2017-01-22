import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import acf, pacf

def m_stat1(df, column):
    df = np.log(df[column])
    df_diff = df - df.shift()
    df_diff.dropna(inplace=True)

    lag_acf = acf(df_diff.values, nlags = 20)
    lag_pacf = pacf(df_diff.values, nlags = 20)
    
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(len(df_diff)),linestyle='--')
    plt.axhline(y=1.96/np.sqrt(len(df_diff)),linestyle='--')   
    
    plt.subplot(121) 
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(len(df_diff)),linestyle='--')
    plt.axhline(y=1.96/np.sqrt(len(df_diff)),linestyle='--')     
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
