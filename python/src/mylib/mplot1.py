import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from matplotlib import style
from pandas.tseries.offsets import *
from mpl_toolkits.mplot3d import Axes3D
from pandas.tools.plotting import radviz
from matplotlib import ticker as mticker
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import bootstrap_plot
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import autocorrelation_plot

def pandas_2d(X, log = False):
    X.plot(logy = log)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
def pandas_scatter_matrix(df):
    scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')

    plt.show()

def pandas_density(df):
    df.plot.kde()

    plt.show()

def pandas_andrews_curves(df, column):
    fig = plt.figure(1)
    
    andrews_curves(df, column)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
def pandas_lag_plot(df, l):
    fig = plt.figure()
    
    func = lag_plot(df, lag=l)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
    return func

def pandas_bootstrap_plot(df):
    bootstrap_plot(df, size=50, samples=500, color='grey')

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def pandas_radviz(df, column):
    radviz(df, column)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def pandas_autocorrelation_plot(df):
    autocorrelation_plot(df)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def pandas_parallel_coordinates(df, column):
    parallel_coordinates(df, column)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def pandas_box(df):
    df.plot.box()
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def pandas_hist(df):
    df.plot.hist(stacked=True, bins=20, orientation='vertical', cumulative=True, alpha=0.5)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def pandas_rolling_std(df, day):
    #pd.rolling_mean(df, window = 2)[1:10] # in future versions you want to resample separately
    r = df.rolling(window = day)
    #r.agg, r.apply, r.count, r.exclusions, r.max, r.median, r.name, r.quantile, r.kurt, r.cov, r.corr, r.aggregate, r.std, r.skew, r.sum, r.var
    #df.plot(style = 'k--')
    r.std().plot(subplots=True, style='k')
    
    plt.show()
