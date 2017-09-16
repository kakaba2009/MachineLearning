import time
import datetime
import numpy as np
import pandas_datareader.data as web
from sqlalchemy import create_engine
import src.mylib.mfile as mfile

Symbols = mfile.loadSymbolList("src/db/source.db")

Symbols = Symbols.values

print(Symbols)

for Symbol in Symbols:
    print(Symbol, "\n")
    start = datetime.datetime(1980, 1, 1)
    end = datetime.datetime(2017, 12, 31)

    try:
        time.sleep(1)
        df = web.get_data_yahoo(Symbol, start, end)
    except Exception as e:
        print(Symbol, e)
        continue
    
    #Covert from panel to data frame
    df = df.to_frame()
    
    siz = len(df["Close"])

    df["Symbol"] = np.repeat(Symbol, siz)
    
    print(df.values, "\n")

    disk_engine = create_engine('sqlite:///src/db/forex.db')

    df.to_sql("FX", disk_engine, if_exists='append')
    #save csv files
    #filename = 'src/db/' + Symbol + '.csv' 
    #df.to_csv(filename, mode='a', columns=['Close'], header=False, index=False)
    #np.savetxt(filename, df.values , fmt='%.2f', delimiter=',')