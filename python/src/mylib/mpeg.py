import numpy as np
import matplotlib
#matplotlib.pyplot.switch_backend('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pandas.tools.plotting import radviz
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import bootstrap_plot
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import autocorrelation_plot

def m_scatter_mpeg(X, Y, filename):
    #FFMpegWriter = manimation.writers['ffmpeg']
    #metadata = dict(title=filename, artist='Matplotlib',comment=filename)
    #writer = FFMpegWriter(fps=15, metadata=metadata)
    
    fig = plt.figure()
    
    func = plt.scatter(X, Y, s=np.abs(Y)*300.0, c=Y, cmap=plt.get_cmap('prism'))

    #with writer.saving(fig, filename, 100):
    for i in range(100):
        func.set_data(X.shift(1), Y.shift(1))
        #writer.grab_frame()
            
def m_scatter_live(X, Y, filename):
    fig, ax = plt.subplots()
    
    ax.scatter(X, Y)
    
    A, B = X, Y

    def run(i):
        print(i)
        
        A = X.shift(i)
        B = Y.shift(i)
        
        ax.clear()
        
        ax.scatter(A, B, s=np.abs(B)*100.0, c=B, cmap=plt.get_cmap('prism'))

    im_ani = animation.FuncAnimation(fig, run, interval=500, repeat_delay=100)
        
    #im_ani.save(filename, metadata={'artist':'Matplotlib'})

    plt.show()
    
def pandas_lag_live(df):
    fig, ax = plt.subplots()
    
    def run(i):
        print(i)
        
        ax.clear()
        
        lag_plot(df, lag=i+1) #lag must be >= 1

    im_ani = animation.FuncAnimation(fig, run, interval=1000, repeat_delay=1000)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
