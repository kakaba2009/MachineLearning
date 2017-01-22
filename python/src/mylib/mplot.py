from matplotlib import style
from matplotlib import ticker as mticker
from matplotlib.finance import candlestick_ohlc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import datetime as dt
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def m_hist2d(x, y, bin, norm):
    plt.figure(1)
    # the histogram of the data
    plt.hist2d(x, y, bin, normed=norm)

    plt.xlabel('Events T')
    plt.ylabel('Events V')
    plt.title('Events Histogram')
    plt.grid(True)
    
    plt.colorbar()

    plt.show()
    
def m_hist(x, bin, norm, c):
    plt.figure(1)
    # the histogram of the data
    n, bins, patches = plt.hist(x, bin, normed=norm, facecolor=c)

    plt.xlabel('Events')
    plt.ylabel('Probability')
    plt.title('Events Histogram')
    plt.grid(True)

    plt.show()

def m_plot2d(X, Y, factor):
    style.use('ggplot')
    
    Y = Y * factor
    
    plt.figure(1)

    plt.plot(X, Y)
    
    plt.tight_layout()
    plt.legend()
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
    return plt

def m_scatter1(X, Y, factor):
    style.use('ggplot')
    
    Y = Y * factor
    
    plt.figure(1)

    plt.scatter(X, Y)
    
    plt.tight_layout()
    plt.legend()
    
    plt.show()    
    
    return plt

def m_scatter2d(X1, Y1):
    plt.figure(1)

    plt.scatter(X1, Y1, label="XY")
    
    plt.tight_layout()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot 2D")
    plt.legend()
    plt.grid(True)

    plt.show()    
    
    return plt

def m_scatter3(X1, Y1, Z1):
    plt.figure(1)

    plt.scatter(X1, Y1, Z1)
    
    plt.tight_layout()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot 3")
    plt.legend()
    plt.grid(True)

    plt.show()    
    
    return plt

def m_scatter2v(X1, Y1, X2, Y2):
    plt.figure(1)

    plt.scatter(X1, Y1, X2, Y2)
    
    plt.tight_layout()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot 2 vectors")
    plt.legend()
    plt.grid(True)

    plt.show()    
    
    return plt

def m_plot(X):
    A = X
        
    style.use('ggplot')

    plt.plot(X)
    
    #ax.xaxis.set_minor_formatter(dates.AutoDateFormatter(dates.AutoDateLocator()))
    #ax.xaxis.grid(True, which="minor")
    #ax.yaxis.grid()
    #ax.xaxis.set_major_formatter(dates.DateFormatter('%b%Y'))

def m_pca(df):
    pca = PCA(n_components=2)

    data = df[["Close","Open"]]

    pca.fit(data.dropna())

    print(pca.explained_variance_ratio_)
    
    return pca

def m_subplot(row, col):
    ax = list()
    
    fig, ax = plt.subplots(row, col, sharex=True, sharey=True)

    return ax

def m_lag3d(df, lag):
    X = df[0    :-2 * lag]
    Y = df[1*lag:-1 * lag]
    Z = df[2*lag:        ]
    
    m_scatter3d(X, Y, Z)

def m_vector(df, lag = 1):
    X = df[0    :-1*lag]
    Y = df[lag  :      ]
    
    U = X[1:]
    V = Y[1:]
    
    return X[:-1], Y[:-1], U, V
            
def m_quiver(df, lag = 1):
    X, Y, U, V = m_vector(df, lag)
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.quiver(X, Y, U, V, color="blue", linewidth=0.1)
    plt.axis('equal')
    plt.grid()
    plt.title('Quive Plot, Dynamic Colours')
    plt.show()                 # display the plot    

def m_arrow(df, lag = 1):
    X = df[0    :-1*lag]
    Y = df[lag  :      ]
    
    draw_path(X.values, Y.values)
    
def distance(X, Y):
    return np.sum((X[1:] - X[:-1]) ** 2 + (Y[1:] - Y[:-1]) ** 2) ** .5

def draw_path(X, Y):
    HEAD_WIDTH = 0.5
    HEAD_LEN   = 0.5

    fig = plt.figure()
    axes = fig.add_subplot(111)

    x = X
    y = Y
    axes.plot(x, y)

    theta = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
    dist  = distance(X, Y) - HEAD_LEN

    x = x[:-1]
    y = y[:-1]
    ax = x + dist * 0.01 * np.sin(theta)
    ay = y + dist * 0.01 * np.cos(theta)

    for x1, y1, x2, y2 in zip(x,y,ax-x,ay-y):
        axes.arrow(x1, y1, x2, y2, head_width=HEAD_WIDTH, head_length=HEAD_LEN)
        
    plt.show()

def m_stream(df, lag = 1):
    X, Y, U, V = m_vector(df, lag)
    
    plot1 = plt.figure()
    plt.streamplot(X, Y, U, V,           # data
               color=U,                  # array that determines the colour
               cmap=plt.get_cmap('cool'),# colour map
               linewidth=2,              # line thickness
               arrowstyle='->',          # arrow style
               arrowsize=1.5)            # arrow size

    #plt.colorbar()                 # add colour bar on the right

    plt.title('Stream Plot, Dynamic Colour')
    plt.show(plot1)                 # display the plot    

def m_scatter3d(X1, Y1, Z1):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X1, Y1, Z1)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.title("Scatter Plot 3D")
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    plt.show()    
    
    return plt

def m_scatter4d(X1, Y1, Z1, C1):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X1, Y1, Z1, c=C1, cmap=plt.get_cmap('prism'))
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.title("Scatter Plot 4D")
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    plt.show()    
    
    return plt

def m_scatter5d(X1, Y1, Z1, S1, C1):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X1, Y1, Z1, s=np.abs(S1), c=C1, cmap=plt.get_cmap('prism'))
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.title("Scatter Plot 5D")
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    plt.show()    
    
    return plt

def m_plot3d(X1, Y1, Z1):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot3D(X1, Y1, Z1, linestyle='None', marker='o')
    #ax.set_prop_cycle
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.title("Plot 3D")
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    plt.show()    
    
def m_scatter_color(X1, Y1, Z1):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    ax.scatter(X1, Y1, s=np.abs(Z1)*10.0, marker='o', c=Y1, cmap=plt.get_cmap('prism'))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    plt.title("Plot Scatter Color")
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    plt.show()

def m_trisurf(X1, Y1, Z1):
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(X1, Y1, Z1, cmap=plt.get_cmap('prism'))
    #ax.set_prop_cycle
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.title("Plot Triple Surface")
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.draw()
    
    plt.show() 

def m_frame(X1, Y1, Z1):
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')

    ax.plot_wireframe(X1, Y1, Z1, cmap=plt.get_cmap('prism'))
    #ax.set_prop_cycle
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.title("Plot wire frame")
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.draw()
    
    plt.show() 
    
def m_image(H):
    fig = plt.figure(figsize=(25, 25))

    ax = fig.add_subplot(1,1,1)
    ax.set_title('colorMap')
    plt.imshow(H, cmap=plt.get_cmap('prism'))
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0.5)
    cax.set_frame_on(False)
    
    plt.colorbar(orientation='vertical')
    plt.show()
    
    return plt

def m_mesh(H):
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    ax.set_title('colorMap')
    
    x, y = H.shape
    X = np.arange(0, x, 1)
    Y = np.arange(0, y, 1)
    X, Y = np.meshgrid(X, Y)
    Z = H[X, Y]
    
    plt.pcolormesh(X, Y, Z)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    
    plt.colorbar(orientation='vertical')
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
    return plt

def m_matrix(M):
    plt.matshow(M, cmap=plt.get_cmap('prism'))
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def m_candlestick(data):
    # convert the datetime64 column in the dataframe to 'float days'
    data['Date']=dates.date2num(data['Date'].astype(dt.date))
    print(data['Date'])

    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    plt.ylabel('Price')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax1.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))

    candlestick_ohlc(ax1,data.values,width=1.0)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
def m_surface(mat):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    matrix = mat #np.array([[0, 1, 2, 3, 4], [.5, 1.5, 2.5, 3.5, 4.5], [1, 2, 3, 4, 5]])
    x, y = matrix.shape
    X = np.arange(0, x, 1)
    Y = np.arange(0, y, 1)
    X, Y = np.meshgrid(X, Y)
    Z = matrix[X, Y]
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap=plt.get_cmap('coolwarm'))

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
