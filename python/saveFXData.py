import datetime
import numpy as np
import pandas_datareader.data as web
from sqlalchemy import create_engine
import src.lib.mfile as mfile

Symbols = mfile.loadSymbolList("src/db/forex.db")

Symbols = Symbols.values

print(Symbols)

for Symbol in Symbols:
    print(Symbol, "\n")
    start = datetime.datetime(1900, 1, 1)
    end = datetime.datetime(2036, 12, 31)

    df = web.get_data_yahoo(Symbol, start, end)
    
    #Covert from panel to data frame
    df = df.to_frame()
    
    siz = len(df["Close"])

    #df.insert(0, "Symbol", Symbol, True)
    df["Symbol"] = np.repeat(Symbol, siz)
    
    print(df, "\n")

    disk_engine = create_engine('sqlite:///src/db/forex1.db')

    df.to_sql("FX", disk_engine, if_exists='append');
