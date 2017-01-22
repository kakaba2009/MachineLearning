import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
from scipy.ndimage.interpolation import shift

def time_delay_embed(array, dimension, time_dif):
    """ A way to generate the time-delay embedding vectors
        for use in later steps, and as the dataset
        array: The name of the series being embedded
        dimension: how many elements to the vector
        time_dif: for if we want to delay by more than one time period (not used for now)"""
    emb = array.values # Converts the panda dataframe to an array
    emb = np.squeeze(np.asarray(emb)) # Make a 1-d array of all values
    i   = len(emb) - 1 # sets up a counter
    new_vec = [] # target for each row
    embed = [] # target for full set
    while i >= dimension-1:
        a = 0  # the dimensional counter
        b = 0  # time_dif counter
        while a< dimension:
            new_vec.append(emb[i-b])
            a+=1
            b+= time_dif
        embed.append(new_vec)
        new_vec = []
        i -=1
        
    X = np.array(embed)
    
    return np.flipud(X)

def vector_delay_embed(array, dimension, time_dif):
    emb = array.values
    emb = np.squeeze(np.asarray(emb))
    emb = np.flipud(emb)
    
    emi = []
    emi.append(emb)
    
    i = 1
    
    while i <= (dimension - 1):
        emi.append(shift(emb, -1 * i * time_dif, cval=np.NaN))
        i +=1
        
    embed = np.stack(emi, axis=-1)
    
    offset = -1 * (dimension - 1) * time_dif
    
    if(offset == 0):
        X = embed
    else:
        X = embed[:offset,]
    
    return np.flipud(X)

def split_x_y(ts):
    X = ts[:-1,  ]
    y = ts[1:,   ]
    
    return X, y

def splitTrainTest(X, y, ratio=0.80):
    location = int(X.shape[0] * ratio)
    
    X_train, X_test = X[0:location , ], X[location: , ]
    y_train, y_test = y[0:location , ], y[location: , ]
    
    return  X_train, X_test, y_train, y_test
    
def m_diff(df, dropna=True):    
    f = lambda x : (2.0 if x > 0.0 else 1.0)

    D = df.diff()
    
    if(dropna == True ):
        D = D.dropna()

    return df, D

def m_pct(df, dropna=True):    
    D = df.pct_change(periods=1, fill_method='pad')

    if(dropna == True ):
        D = D.dropna()

    return D

def m_log(df, dropna=True):    
    df = df.apply(np.log)

    if(dropna == True ):
        df = df.dropna()

    return df

def m_log_diff(df, dropna=True):    
    D = df.apply(np.log)
    
    D = D.diff(periods=1)

    D = D.dropna()
    
    print(D.describe())

    return df, D

def m_sample_base(df, feq):
    if(df.index.name != "Date") :
        df = df.set_index("Date", drop=False)

    ohlc_dict = {
                 'Date': 'last',
                 'Open': 'first',
                 'High': 'max',
                 'Low':  'min',
                 'Close':'last'
    }

    o = df["Open" ].resample(feq, axis=0).first()
    h = df["High" ].resample(feq, axis=0).max()
    l = df["Low"  ].resample(feq, axis=0).min()
    c = df["Close"].resample(feq, axis=0).last()
    
    o = o.fillna(method='backfill')
    h = h.fillna(method='backfill')
    l = l.fillna(method='backfill')
    c = c.fillna(method='backfill')

    X = pd.concat([o, h, l, c], axis=1, join_axes=[o.index])
    
    print(X.describe())
    
    na = X.isnull().sum()

    #print("nan count:\n", na)
    
    return X

def m_sample_w(df):
    X = m_sample_base(df, "W")
    
    print(X.columns)
    
    return X

def m_sample_m(df):
    X = m_sample_base(df, BMonthEnd())
    
    print(X.columns)
    
    return X

def m_sample_q(df):
    X = m_sample_base(df, BQuarterEnd())
    
    print(X.columns)
    
    return X

def m_sample_y(df):
    X = m_sample_base(df, BYearEnd())
    
    print(X.columns)
    
    return X

def c_mesh(dfs, symbols):
    mat = np.array([])
    
    for s in symbols:
        df = dfs.get_group(s).iloc[1:100][["Open", "High", "Low", "Close"]]
        
        if(mat.size == 0):
            mat = df.values
        else:
            mat = np.hstack((mat, df))
    
    return mat

def c_cumsum(df):
    df = df.cumsum(axis=1)
    
    return df

def c_qcut(df, q, label):
    qc, bins  = pd.qcut(df, q, retbins=True, labels=label)

    return qc, bins

def c_value_counts(df, bin):
    rv = df.value_counts(bins=bin)
    
    return rv

def c_lastn(df, n):
    return df[-1 * n:]
