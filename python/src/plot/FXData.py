import src.lib.mfile as mfile
import src.lib.mcalc as mcalc
import src.lib.mplot as mplot

df = mfile.loadSymbolList("../db/forex.db")
df = df["Symbol"]
df = mfile.loadOneSymbol("JPY=X", "../db/forex.db")
df = df[-1000:]
#df = m_sample_m(df)
df = df[['High']]
#df = dfs.groupby("Symbol")
#df = dfs.get_group("JPY=X").drop("Symbol",axis=1)
#df = m_pct(df, dropna=True)
X  = mcalc.vector_delay_embed(df, 2, 10)
#Y["SDate"] = Y.index.year + Y.index.dayofyear / 365.0
#W["SDate"] = W.index.year + W.index.dayofyear / 365.0
#W["SDate"] = mdates.date2num(W.index.to_datetime())

#pandas_lag_plot(df.High, 32)
#m_lag3d(df.High, 32)
a = X[:,0]
b = X[:,1]
#c = X[:,2]
#d = X[:,3]
#e = X[:,4]
mplot.m_scatter2d(a, b)
