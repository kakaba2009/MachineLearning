import datetime
import numpy as np
import pandas_datareader.data as web
from sqlalchemy import create_engine
import src.mylib.mfile as mfile

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

    df["Symbol"] = np.repeat(Symbol, siz)
    
    print(df.values, "\n")

    disk_engine = create_engine('sqlite:///src/db/forex1.db')

    df.to_sql("FX", disk_engine, if_exists='append');
    
    #save csv files
    #filename = 'src/db/' + Symbol + '.csv' 
    #df.to_csv(filename, mode='a', columns=['Close'], header=False, index=False)
    #np.savetxt(filename, df.values , fmt='%.2f', delimiter=',')